"""Exporta un caso listo para el dashboard (Semana 11): corre el pipeline una vez y guarda todo.

El dashboard no infiere el volumen completo en vivo (200 a 400 cortes tardan minutos en CPU);
solo lee lo que este script dejó en ``outputs/demo/<modelo>/<caso>/``:

    volume.npz   images [z,S,S] uint8, semantic, ids (volumen de IDs PENGWIN predicho), gt_ids opcional
    meshes.npz   mallas por fragmento (predicho ``p_<id>_v/_f`` y real ``g_<id>_v/_f``)
    raw_view.npz MIP axial/coronal/sagital e isosuperficie del hueso crudo (visualizador 1)
    meta.json    spacing, cajas por corte, tabla de fragmentos con distancia en mm, comparación con GT

Uso (desde la raíz del repo, con los datos en data/raw):

    python src/demo_export.py --case 024 044 099 --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth --tta

ORIENTACIÓN. H6 se entrenó con los volúmenes en orientación canónica (``orientacion.py``): los .mha
guardados en LPI (girados 180 grados en el plano) se enderezan antes de predecir. Si el checkpoint lo
pide (``model.canonical``), el caso se endereza UNA vez al cargarlo (imagen, ground truth y vista
cruda), así los tres visualizadores muestran la misma orientación y la inferencia en vivo recibe
cortes ya enderezados. Si un volumen está girado se decide con la cabecera del .mha, no con la
lista fija de PENGWIN. D5 se entrenó sin ese paso: con D5 no se endereza. ``meta.json`` guarda
``canonical`` (si el modelo lo pide) y ``flipped`` (si este caso se giró).

DISTANCIA EN MM. ``postprocess_3d(fill_unlabeled=False)``: los vóxeles de la región que no son ni
principal ni fragmento quedan sin etiqueta, así no se rellena el hueco entre fragmentos (ver
docs/decisiones_semana10.md, sección 5). Es el mismo método de ``infer_volume.py --distance-nofill``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fragment_distances import REGIONS, measure_fragment_distances
from volume_postprocess import match_fragments, postprocess_3d

REGION_LABELS = {"sacro": "Sacro", "coxal_izquierdo": "Coxal izquierdo", "coxal_derecho": "Coxal derecho"}
REGION_STARTS = {"sacro": 1, "coxal_izquierdo": 11, "coxal_derecho": 21}
DISTANCE_METHOD = "postprocess_3d(fill_unlabeled=False)"


def region_of(fragment_id: int) -> str:
    for name, (low, high, _) in REGIONS.items():
        if low <= fragment_id <= high:
            return name
    raise ValueError(f"ID fuera de rango: {fragment_id}")


def build_id_volume(semantic: np.ndarray, primary_secondary: np.ndarray,
                    min_voxels: int = 30) -> tuple[np.ndarray, str]:
    """Volumen de IDs PENGWIN a partir de las predicciones, sin el relleno que pega fragmentos.

    Devuelve (volumen, descripción del método). Lo que se ve en los visualizadores es el mismo
    volumen sobre el que se mide la distancia.
    """
    return postprocess_3d(semantic, primary_secondary, min_voxels, fill_unlabeled=False), DISTANCE_METHOD


def fragment_table(volume: np.ndarray, spacing_xyz: tuple[float, float, float]) -> list[dict]:
    """Una fila por fragmento presente: región, ID, vóxeles, volumen en mm³ y distancia al principal."""
    voxel_mm3 = float(np.prod(spacing_xyz))
    distances = {item.fragment_label: item.distance_mm
                 for item in measure_fragment_distances(volume, spacing_xyz)}
    rows = []
    ids, counts = np.unique(volume[volume > 0], return_counts=True)
    for fragment_id, count in zip(ids.tolist(), counts.tolist()):
        region = region_of(fragment_id)
        is_main = fragment_id == REGIONS[region][2]
        coords = np.argwhere(volume == fragment_id)
        centroid_zyx = coords.mean(axis=0)
        rows.append({"id": int(fragment_id), "region": region, "is_main": bool(is_main),
                     "voxels": int(count), "volume_mm3": float(count * voxel_mm3),
                     "distance_mm": None if is_main else float(distances[fragment_id]),
                     "centroid_zyx_vox": [float(v) for v in centroid_zyx],
                     "z_range": [int(coords[:, 0].min()), int(coords[:, 0].max())]})
    return rows


def compare_tables(pred_rows: list[dict], gt_rows: list[dict], prediction: np.ndarray,
                   truth: np.ndarray) -> list[dict]:
    """Por cada fragmento secundario real: su distancia, el fragmento predicho que lo cubre y el error."""
    pred = {r["id"]: r for r in pred_rows}
    matches = {m["gt_id"]: m for m in match_fragments(prediction, truth)}
    rows = []
    for gt in gt_rows:
        if gt["is_main"]:
            continue
        match = matches.get(gt["id"], {})
        pred_id = match.get("pred_id")
        pred_row = pred.get(pred_id) if pred_id is not None else None
        pred_mm = pred_row["distance_mm"] if pred_row and not pred_row["is_main"] else None
        rows.append({"region": gt["region"], "gt_id": gt["id"], "gt_mm": gt["distance_mm"],
                     "pred_id": pred_id, "pred_mm": pred_mm,
                     "abs_error_mm": None if pred_mm is None else abs(pred_mm - gt["distance_mm"]),
                     "dice3d": match.get("dice", 0.0), "iou3d": match.get("iou", 0.0),
                     "voxels_gt": gt["voxels"]})
    return rows


def fragment_meshes(volume: np.ndarray, spacing_xyz: tuple[float, float, float], step: int = 2,
                    min_voxels: int = 100) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Marching cubes por fragmento. Vértices (z,y,x) en mm; ``step`` reduce los triángulos."""
    sx, sy, sz = spacing_xyz
    spacing_zyx = (sz, sy, sx)
    meshes = {}
    for fragment_id in (int(v) for v in np.unique(volume) if v > 0):
        mask = volume == fragment_id
        if int(mask.sum()) < min_voxels:
            continue
        # Los fragmentos grandes (los principales) tienen cientos de miles de triángulos con paso 2;
        # con paso 3 se ven igual de bien y la malla abre en un celular.
        step_here = step + 1 if int(mask.sum()) > 50_000 else step
        coords = np.argwhere(mask)
        lo = np.maximum(coords.min(axis=0) - 2, 0)
        hi = coords.max(axis=0) + 3
        crop = mask[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
        raw_field = np.pad(crop.astype(np.float32), 1)
        result = None
        # Del más suave al más crudo: un fragmento muy fino puede no cruzar 0.5 tras suavizar
        # o tras saltar vóxeles con ``step_size``.
        for field, step_try in ((np.pad(ndi.gaussian_filter(crop.astype(np.float32), 0.8), 1), step_here),
                                (raw_field, step_here), (raw_field, 1)):
            try:
                result = measure.marching_cubes(field, level=0.5, spacing=spacing_zyx, step_size=step_try)
                break
            except (RuntimeError, ValueError):
                continue
        if result is None:
            continue
        verts, faces = result[0], result[1]
        verts = verts - np.asarray(spacing_zyx) + lo * np.asarray(spacing_zyx)
        meshes[fragment_id] = (verts.astype(np.float32), faces.astype(np.int32))
    return meshes


def checkpoint_is_canonical(checkpoint: str | Path) -> bool:
    """Lo mismo que deja ``evaluate_volumes.load_model`` en ``model.canonical``, sin cargar el modelo."""
    import torch
    return bool(torch.load(checkpoint, map_location="cpu").get("canonical", False))


def checkpoint_uses_normalization(checkpoint: str | Path) -> bool:
    """Indica si el entrenamiento del checkpoint usó enderezado pélvico."""
    import torch
    return bool(torch.load(checkpoint, map_location="cpu").get("normalizar", False))


def export_case(case_id: str, checkpoint: str, out_root: str = "outputs/demo", size: int = 384,
                tta: bool = True, batch_size: int = 8, min_voxels: int = 30, raw_view: bool = True,
                reuse: bool = False, normalizar: bool | None = None, progress=None) -> Path:
    """Corre el pipeline sobre un caso y escribe la carpeta del dashboard. Devuelve su ruta.

    Con ``reuse=True`` y una exportación previa del mismo caso, no vuelve a correr el modelo: lee
    ``predictions.npz`` y rehace solo el post-proceso, las distancias y las mallas. Es lo que se usa
    cuando cambia el post-proceso 3D.
    """
    from orientacion import (canonicalize_arrays, estimate_pelvis_correction,
                             rotate_inplane_physical)
    from pengwin_io import load_case, resize_label_slice
    from predict_volume import (cache_spacing_xyz, load_pipeline, predict_volume,
                                preprocess_volume)

    case_id = str(case_id).zfill(3)
    if normalizar is None:
        normalizar = checkpoint_uses_normalization(checkpoint)
    if reuse and normalizar:
        raise ValueError("No combines reuse=True con normalizar=True: las predicciones previas pueden no estar giradas")
    log = progress or (lambda message: print(message, flush=True))
    started = time.perf_counter()
    case = load_case(case_id, load_label=True)
    canonical = checkpoint_is_canonical(checkpoint)
    flipped = False
    orientation = {"applied": False, "method": "disabled"}
    if canonical:
        case.image, case.label, case.spacing_xyz, orientation = canonicalize_arrays(
            case.image, case.label, case.direction_xyz, case.spacing_xyz)
        flipped = bool(orientation["applied"])
        case.direction_xyz = tuple(np.eye(3).reshape(-1))
        log(f"[{case_id}] orientación por metadatos: "
            f"{'corregida' if orientation['applied'] else 'ya canónica'} (punto/cruz)")
    elif normalizar:
        raise ValueError("--normalizar requiere un checkpoint entrenado con orientación canónica")
    spacing = cache_spacing_xyz(case.spacing_xyz, case.image.shape, size)
    log(f"[{case_id}] volumen {case.image.shape} spacing nativo {tuple(round(float(s), 3) for s in case.spacing_xyz)} mm")

    rotation = {"accepted": False, "reason": "disabled", "angle_degrees": 0.0}
    if normalizar:
        preview = preprocess_volume(case.image, size)
        rotation = estimate_pelvis_correction(preview, spacing[1], spacing[0])
        if rotation["accepted"]:
            case.image = rotate_inplane_physical(
                case.image, rotation["angle_degrees"], case.spacing_xyz[1], case.spacing_xyz[0],
                order=1, cval=-1024)
            if case.label is not None:
                case.label = rotate_inplane_physical(
                    case.label, rotation["angle_degrees"], case.spacing_xyz[1], case.spacing_xyz[0], order=0)
            log(f"[{case_id}] giro anatómico corregido {rotation['angle_degrees']:+.1f}° "
                f"(mejora de simetría {rotation['improvement']:.3f})")
        else:
            log(f"[{case_id}] giro anatómico sin aplicar: {rotation['reason']} "
                f"(estimado {rotation['angle_degrees']:+.1f}°)")

    out_dir = Path(out_root) / Path(checkpoint).stem / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = out_dir / "predictions.npz"
    if reuse and saved.exists() and (out_dir / "meta.json").exists():
        old = json.loads((out_dir / "meta.json").read_text())
        with np.load(saved) as arrays, np.load(out_dir / "volume.npz") as volume:
            pred = {"semantic": arrays["semantic"], "primary_secondary": arrays["primary_secondary"],
                    "classification": np.array([s["classification"] for s in old["slices"]], dtype=np.float32),
                    "boxes_raw": [s["boxes_raw"] for s in old["slices"]],
                    "boxes_refined": [s["boxes_refined"] for s in old["slices"]],
                    "seconds": old["seconds_model"]}
            images = volume["images"]
        info, device, tta = old["model"], old["device"], old["tta"]
        log(f"[{case_id}] predicciones reutilizadas de {saved.name} (el modelo no se vuelve a correr)")
    else:
        images = preprocess_volume(case.image, size)
        model, device, info = load_pipeline(checkpoint, tta=tta)
        assert info["canonical"] == canonical, "el checkpoint y el modelo cargado no coinciden en orientación"
        log(f"[{case_id}] modelo {info['checkpoint']} ({info['parameters'] / 1e6:.2f} M parámetros) en {device}, TTA={tta}")
        pred = predict_volume(model, images, device, batch_size=batch_size,
                              progress=lambda done, total: log(f"[{case_id}] cortes {done}/{total}")
                              if done == total or done % (batch_size * 10) == 0 else None)
        log(f"[{case_id}] inferencia: {pred['seconds']:.1f} s ({1000 * pred['seconds'] / len(images):.0f} ms por corte)")
        np.savez_compressed(saved, semantic=pred["semantic"], primary_secondary=pred["primary_secondary"])

    ids, method = build_id_volume(pred["semantic"], pred["primary_secondary"], min_voxels)
    t = time.perf_counter()
    rows = fragment_table(ids, spacing)
    log(f"[{case_id}] distancias predichas ({len(rows)} fragmentos) en {time.perf_counter() - t:.1f} s")

    gt_ids = gt_rows = comparison = None
    if case.label is not None:
        gt_ids = np.stack([resize_label_slice(case.label[z].astype(np.uint8), size)
                           for z in range(len(images))])
        gt_rows = fragment_table(gt_ids, spacing)
        comparison = compare_tables(rows, gt_rows, ids, gt_ids)
        log(f"[{case_id}] ground truth: {len(gt_rows)} fragmentos, {len(comparison)} secundarios")

    arrays = {"images": images, "semantic": pred["semantic"], "ids": ids}
    if gt_ids is not None:
        arrays["gt_ids"] = gt_ids
    np.savez_compressed(out_dir / "volume.npz", **arrays)

    mesh_arrays = {}
    for prefix, volume in (("p", ids), ("g", gt_ids)):
        if volume is None:
            continue
        for fragment_id, (verts, faces) in fragment_meshes(volume, spacing).items():
            mesh_arrays[f"{prefix}_{fragment_id}_v"] = verts
            mesh_arrays[f"{prefix}_{fragment_id}_f"] = faces
    np.savez_compressed(out_dir / "meshes.npz", **mesh_arrays)
    log(f"[{case_id}] mallas: {sum(len(v) for k, v in mesh_arrays.items() if k.endswith('_f')):,} triángulos")

    if raw_view:
        from raw_view import bone_mask_for, bone_mesh, compute_mips
        mask = bone_mask_for(case.image, case.spacing_xyz)
        mips = compute_mips(case.image, mask, case.spacing_xyz)
        mesh = bone_mesh(mask, case.spacing_xyz)
        np.savez_compressed(out_dir / "raw_view.npz", mip_axial=mips["axial"], mip_coronal=mips["coronal"],
                            mip_sagital=mips["sagital"], bone_verts_zyx=mesh["verts_zyx"],
                            bone_faces=mesh["faces"], bone_factor=mesh["factor"])
        log(f"[{case_id}] visualizador 1: MIP y malla de {len(mesh['faces']):,} triángulos")

    bone_z = np.flatnonzero((pred["semantic"] > 0).any(axis=(1, 2)))
    meta = {"case_id": case_id, "model": info, "size": size, "n_slices": int(len(images)),
            "spacing_cache_xyz": list(spacing), "spacing_native_xyz": [float(s) for s in case.spacing_xyz],
            "orig_shape_zyx": [int(s) for s in case.image.shape], "tta": tta,
            "device": str(device), "seconds_model": pred["seconds"],
            "ms_per_slice": 1000 * pred["seconds"] / len(images), "min_voxels": min_voxels,
            "distance_method": method, "canonical": canonical, "flipped": flipped,
            "orientation_normalization": orientation, "patient_rotation_normalization": rotation,
            "bone_z_range": [int(bone_z[0]), int(bone_z[-1])] if len(bone_z) else None,
            "fragments_pred": rows, "fragments_gt": gt_rows, "comparison": comparison,
            "slices": [{"z": z, "classification": [round(float(p), 4) for p in pred["classification"][z]],
                        "boxes_raw": pred["boxes_raw"][z], "boxes_refined": pred["boxes_refined"][z]}
                       for z in range(len(images))]}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"[{case_id}] listo en {time.perf_counter() - started:.0f} s → {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", default="outputs/demo")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--tta", action="store_true", help="Espejo izquierda-derecha, como en la validación.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-voxels", type=int, default=30)
    parser.add_argument("--no-raw-view", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="No corre el modelo: reutiliza predictions.npz de una exportación previa.")
    parser.add_argument("--normalizar", action=argparse.BooleanOptionalAction, default=None,
                        help="Activa/desactiva el enderezado; por defecto usa la configuración del checkpoint.")
    args = parser.parse_args()
    for case_id in args.case:
        export_case(case_id, args.checkpoint, args.out_root, args.size, args.tta, args.batch_size,
                    args.min_voxels, raw_view=not args.no_raw_view, reuse=args.reuse,
                    normalizar=args.normalizar)


if __name__ == "__main__":
    main()
