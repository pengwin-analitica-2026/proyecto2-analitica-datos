"""Exporta un caso listo para el dashboard (Semana 11): corre el pipeline una vez y guarda todo.

El dashboard no infiere el volumen completo en vivo (200 a 400 cortes tardan minutos en CPU);
solo lee lo que este script dejó en ``outputs/demo/<modelo>/<caso>/``:

    volume.npz   images [z,S,S] uint8, semantic, ids (volumen de IDs PENGWIN predicho), gt_ids opcional
    meshes.npz   mallas por fragmento (predicho ``p_<id>_v/_f`` y real ``g_<id>_v/_f``)
    raw_view.npz MIP axial/coronal/sagital e isosuperficie del hueso, cruda y pulida (visualizador 1)
    meta.json    spacing, cajas por corte, tabla de fragmentos con distancia en mm, comparación con GT

Uso (desde la raíz del repo, con los datos en data/raw):

    python src/demo_export.py --case 024 044 099 --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth --tta

ORIENTACIÓN. H8 (como H6 y H7) se entrenó con los volúmenes en orientación canónica (``orientacion.py``): los .mha
guardados en LPI (girados 180 grados en el plano) se enderezan antes de predecir. Si el checkpoint lo
pide (``model.canonical``), el caso se endereza UNA vez al cargarlo (imagen, ground truth y vista
cruda), así los tres visualizadores muestran la misma orientación y la inferencia en vivo recibe
cortes ya enderezados. Si un volumen está girado se decide con la cabecera del .mha, no con la
lista fija de PENGWIN. D5 se entrenó sin ese paso: con D5 no se endereza. ``meta.json`` guarda
``canonical`` (si el modelo lo pide) y ``flipped`` (si este caso se giró).

ENDEREZADO DEL GIRO AXIAL. Después de la orientación, si el paciente está girado dentro del escáner
(algo que la cabecera no dice), ``normalizacion.normalizar`` estima el ángulo con la simetría
izquierda-derecha de la pelvis vista desde arriba y gira el volumen y el ground truth para
enderezarlo. Solo corrige giros claros, de al menos ``GIRO_MINIMO_DEMO`` grados: así un paciente
bien acostado queda igual. ``meta.json`` guarda el ángulo estimado y si se corrigió. Con
``--sin-enderezar`` se omite.

UMBRALES EN MM³. Los tamaños mínimos del post-proceso 3D (fragmento y región) se fijan en mm³ con el
espaciado del caso (``volume_postprocess.umbrales_en_voxeles``), así no dependen del grosor de los
cortes. Con ``--umbral-voxeles`` se usan los 30 y 200 vóxeles fijos de la evaluación oficial.

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
from calidad_entrada import aviso_metal, marcar_fragmentos_con_metal, revisar_espaciado, revisar_metal
from fragment_distances import REGIONS, measure_fragment_distances
from volume_postprocess import match_fragments, postprocess_3d, umbrales_en_voxeles

REGION_LABELS = {"sacro": "Sacro", "coxal_izquierdo": "Coxal izquierdo", "coxal_derecho": "Coxal derecho"}
REGION_STARTS = {"sacro": 1, "coxal_izquierdo": 11, "coxal_derecho": 21}
DISTANCE_METHOD = "postprocess_3d(fill_unlabeled=False)"


# Umbral del enderezado en la demo. Con 10° no cambia ningún caso de validación sin girar (los dos
# que se corregían con 5°, 014 y 095, tenían 6°) y los giros de 30° y 60° se siguen corrigiendo.
GIRO_MINIMO_DEMO = 10.0


def region_of(fragment_id: int) -> str:
    for name, (low, high, _) in REGIONS.items():
        if low <= fragment_id <= high:
            return name
    raise ValueError(f"ID fuera de rango: {fragment_id}")


def build_id_volume(semantic: np.ndarray, primary_secondary: np.ndarray, min_voxels: int = 30,
                    spacing_xyz: tuple[float, float, float] | None = None) -> tuple[np.ndarray, str]:
    """Volumen de IDs PENGWIN a partir de las predicciones, sin el relleno que pega fragmentos.

    Con ``spacing_xyz`` los umbrales de tamaño se fijan en mm³ (``umbrales_en_voxeles``) y se
    adaptan al grosor de los cortes; sin él se usan ``min_voxels`` y 200 vóxeles. Devuelve
    (volumen, descripción del método). Lo que se ve en los visualizadores es el mismo volumen sobre
    el que se mide la distancia.
    """
    min_region = 200
    if spacing_xyz is not None:
        min_voxels, min_region = umbrales_en_voxeles(spacing_xyz)
    return (postprocess_3d(semantic, primary_secondary, min_voxels, min_region, fill_unlabeled=False),
            DISTANCE_METHOD)


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
        # Solo para la malla: rellena las cavidades cerradas dentro de este fragmento. Va fragmento por
        # fragmento, así que no une fragmentos, y las distancias y tablas ya se midieron sobre ``volume``.
        crop = ndi.binary_fill_holes(np.pad(crop, 1))[1:-1, 1:-1, 1:-1]
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


def _caja_hueso(*volumenes, margen: int = 2) -> tuple[slice, slice, slice]:
    """Caja (z, y, x) que contiene todo lo distinto de cero de los volúmenes dados, con un margen."""
    lo, hi = None, None
    for volumen in volumenes:
        if volumen is None or not volumen.any():
            continue
        ejes = [np.flatnonzero(volumen.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
        a = np.array([e[0] for e in ejes]); b = np.array([e[-1] + 1 for e in ejes])
        lo, hi = (a, b) if lo is None else (np.minimum(lo, a), np.maximum(hi, b))
    forma = next(v.shape for v in volumenes if v is not None)
    if lo is None:
        return tuple(slice(0, n) for n in forma)
    return tuple(slice(max(int(a) - margen, 0), min(int(b) + margen, n)) for a, b, n in zip(lo, hi, forma))


def _desplazar(rows: list[dict], origen: np.ndarray) -> list[dict]:
    """Lleva centroides y rangos de cortes de la caja a coordenadas del volumen completo."""
    for row in rows:
        row["centroid_zyx_vox"] = [float(c + o) for c, o in zip(row["centroid_zyx_vox"], origen)]
        row["z_range"] = [int(z + origen[0]) for z in row["z_range"]]
    return rows


def checkpoint_is_canonical(checkpoint: str | Path) -> bool:
    """Lo mismo que deja ``evaluate_volumes.load_model`` en ``model.canonical``, sin cargar el modelo."""
    import torch
    return bool(torch.load(checkpoint, map_location="cpu").get("canonical", False))


def predictor_local(checkpoint: str | Path, tta: bool = True, batch_size: int = 8,
                    log=print, precision: str = "fp32") -> tuple[dict, object]:
    """Modelo local con torch. Devuelve (info del modelo, función que predice un volumen uint8).

    ``precision``: fp32, int8 (CPU), fp16 (GPU) o auto; ver ``precision.py``.
    """
    from predict_volume import load_pipeline, predict_volume
    model, device, info = load_pipeline(checkpoint, tta=tta, precision=precision)
    info = dict(info, device=str(device))

    def predecir(images: np.ndarray, progress=None) -> dict:
        return predict_volume(model, images, device, batch_size=batch_size, progress=progress)
    return info, predecir


def _vista_cruda(image: np.ndarray, spacing_xyz, out_dir: Path, log, tag: str,
                 max_voxeles: int | None = None) -> None:
    """Visualizador 1: MIP en tres ejes e isosuperficies del hueso, sin modelo.

    Con ``max_voxeles`` (el dashboard publicado, con poca memoria) un volumen más grande se reduce en el
    plano (cada 2 o más píxeles) solo para esta vista: la limpieza de la máscara ósea trabaja sobre el
    volumen completo y con 58 millones de vóxeles pasaba de 1.6 GB. Es solo visual: las MIP ya se
    llevan a 1 mm y la malla ya se reduce; el modelo, las distancias y las métricas no usan esta vista.
    """
    import gc
    from raw_view import bone_mask_for, bone_mesh, compute_mips
    factor = 1
    while max_voxeles and image[:, ::factor, ::factor].size > max_voxeles and factor < 8:
        factor += 1
    if factor > 1:
        image = image[:, ::factor, ::factor]
        spacing_xyz = (float(spacing_xyz[0]) * factor, float(spacing_xyz[1]) * factor, float(spacing_xyz[2]))
        log(f"{tag} vista cruda reducida ×{factor} en el plano (volumen grande, solo visual)")
    mask = bone_mask_for(image, spacing_xyz)
    mips = compute_mips(image, mask, spacing_xyz)          # las MIP usan la máscara cruda
    mesh = bone_mesh(mask, spacing_xyz)
    del mask
    gc.collect()
    # Malla pulida: misma limpieza más la recuperación de hueso tenue. Solo visual.
    mesh_p = bone_mesh(bone_mask_for(image, spacing_xyz, pulido=True), spacing_xyz)
    np.savez_compressed(out_dir / "raw_view.npz", mip_axial=mips["axial"], mip_coronal=mips["coronal"],
                        mip_sagital=mips["sagital"], bone_verts_zyx=mesh["verts_zyx"],
                        bone_faces=mesh["faces"], bone_factor=mesh["factor"],
                        bone_verts_zyx_pulido=mesh_p["verts_zyx"], bone_faces_pulido=mesh_p["faces"],
                        bone_factor_pulido=mesh_p["factor"])
    log(f"{tag} visualizador 1: MIP y mallas de {len(mesh['faces']):,} (cruda) y "
        f"{len(mesh_p['faces']):,} (pulida) triángulos")


def exportar_volumen(image: np.ndarray, spacing_xyz, out_dir: str | Path, info: dict, predecir,
                     label: np.ndarray | None = None, nombre: str = "caso", size: int = 384,
                     min_voxels: int = 30, raw_view: bool = True, enderezar: bool = True,
                     umbral_mm: bool = True, flipped: bool = False, extra_meta: dict | None = None,
                     pred_guardada: dict | None = None, pasos=None, log=print,
                     max_voxeles_vista: int | None = None, reducir_metal: bool = False) -> Path:
    """Pipeline completo sobre un volumen ya cargado y en la orientación que espera el modelo.

    ``image`` (z,y,x) en HU y ``label`` opcional, ya orientados. ``info`` describe el modelo
    (``checkpoint``, ``parameters``, ``tta``, ``device``, ``canonical``) y ``predecir(images_u8,
    progress)`` devuelve lo mismo que ``predict_volume``: así el modelo puede ser local o la API de
    AWS. Con ``pred_guardada`` (y ``info`` de la exportación anterior) no se vuelve a predecir.
    ``pasos`` (``pasos_preproceso.Pasos``) trae las vistas de las etapas previas (crudo y orientado);
    aquí se agregan el giro axial y la entrada al modelo, y se guardan en ``pasos.npz``.
    Escribe en ``out_dir`` los archivos que lee el dashboard y devuelve esa carpeta.
    """
    from pasos_preproceso import Pasos
    from volumen_entrada import cache_spacing_xyz, preprocess_volume
    from pengwin_io import resize_label_slice

    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{nombre}]"
    pasos = pasos or Pasos()
    reduccion = None
    if reducir_metal:
        # Experimental (reduccion_metal.py): atenúa las rayas del metal antes del modelo. Va antes del enderezado
        # porque trabaja corte a corte sobre los HU tal como vienen.
        from reduccion_metal import reducir_artefactos_metal
        image, reduccion = reducir_artefactos_metal(image, log=lambda texto: log(f"{tag} {texto}"))
        pasos.agregar_hu("metal", "Reducción de artefactos de metal (experimental)",
                         (f"LI-MAR en {len(reduccion['cortes'])} cortes con metal ({reduccion['segundos']:.0f} s): se "
                          "reemplazan los rayos que pasan por el metal por una interpolación de los vecinos y se "
                          "reconstruye el corte; el metal se repone tal cual." if reduccion["cortes"] else
                          "No hay cortes con metal: el volumen no cambia."), image, spacing_xyz)
    giro = None
    if enderezar:
        from normalizacion import normalizar
        image, label, est = normalizar(image, spacing_xyz, label, giro_minimo=GIRO_MINIMO_DEMO)
        giro = {"angulo": round(float(est.angulo), 2), "corregido": bool(est.corregir),
                "giro_minimo": GIRO_MINIMO_DEMO, "simetria_antes": round(float(est.simetria_sin_girar), 3),
                "simetria_despues": round(float(est.simetria), 3)}
        log(f"{tag} giro axial estimado {est.angulo:+.1f}°: "
            + ("enderezado" if est.corregir else "sin corregir (menor al umbral o no es claro)"))
        texto_giro = (f"Giro del paciente dentro del escáner estimado en {est.angulo:+.1f}° con la simetría "
                      "izquierda-derecha de la pelvis (no está en los metadatos). "
                      + (f"Es claro (≥ {GIRO_MINIMO_DEMO:.0f}°): se enderezó girando el volumen {-est.angulo:+.1f}°."
                         if est.corregir else f"Menor a {GIRO_MINIMO_DEMO:.0f}° o no es claro: no se corrigió, "
                         "la imagen queda igual que en el paso anterior."))
    else:
        texto_giro = "El enderezado del giro axial está desactivado: la imagen queda igual que en el paso anterior."
    pasos.agregar_hu("giro_axial", "Giro axial del paciente", texto_giro, image, spacing_xyz, giro=giro)
    spacing = cache_spacing_xyz(spacing_xyz, image.shape, size)
    log(f"{tag} volumen {image.shape} spacing nativo {tuple(round(float(s), 3) for s in spacing_xyz)} mm")

    images = preprocess_volume(image, size)
    # HU originales del corte (ya orientado y enderezado), a la misma resolución que ``images``: solo para que
    # los visualizadores dejen elegir cualquier ventana de grises, de −1024 HU al máximo del volumen (metal).
    from pengwin_io import resize_slice
    hu_vista = np.stack([np.rint(resize_slice(np.clip(corte, -1024, 32000).astype(np.float32), size))
                         for corte in image]).astype(np.int16)
    pasos.agregar_entrada("Entrada al modelo", f"Ventana ósea (−500 a 1300 HU), gamma 1.5 y cada corte estirado "
                          f"a {size}×{size} píxeles: exactamente lo que recibe la red, corte por corte.", images)
    np.savez_compressed(out_dir / "pasos.npz", **pasos.arreglos)
    if pred_guardada is not None:
        pred = pred_guardada
        log(f"{tag} predicciones reutilizadas (el modelo no se vuelve a correr)")
    else:
        log(f"{tag} modelo {info['checkpoint']} ({info['parameters'] / 1e6:.2f} M parámetros) en "
            f"{info.get('device')}, TTA={info.get('tta')}")
        pred = predecir(images, lambda done, total: log(f"{tag} cortes {done}/{total}")
                        if done == total or done % 80 == 0 else None)
        log(f"{tag} inferencia: {pred['seconds']:.1f} s ({1000 * pred['seconds'] / len(images):.0f} ms por corte)")
        np.savez_compressed(out_dir / "predictions.npz", semantic=pred["semantic"],
                            primary_secondary=pred["primary_secondary"])

    gt_ids = None
    if label is not None:
        gt_ids = np.stack([resize_label_slice(label[z].astype(np.uint8), size)
                           for z in range(len(images))])
    # Memoria: el post-proceso 3D, las distancias y las mallas se calculan dentro de la caja que contiene
    # el hueso (predicho y real), no sobre todo el volumen a 384 px. El resultado es idéntico (fuera de la
    # caja todo es fondo) y en un CT grande la memoria baja a la mitad: el dashboard publicado tiene ~2.7 GB.
    caja = _caja_hueso(pred["semantic"], pred["primary_secondary"], gt_ids)
    origen = np.array([c.start for c in caja])
    ids_caja, method = build_id_volume(pred["semantic"][caja], pred["primary_secondary"][caja], min_voxels,
                                       spacing_xyz=spacing if umbral_mm else None)
    ids = np.zeros(pred["semantic"].shape, dtype=np.uint8)
    ids[caja] = ids_caja
    umbrales = {"modo": "mm3" if umbral_mm else "voxeles",
                "min_voxels": umbrales_en_voxeles(spacing)[0] if umbral_mm else min_voxels,
                "min_region_voxels": umbrales_en_voxeles(spacing)[1] if umbral_mm else 200}
    t = time.perf_counter()
    rows = _desplazar(fragment_table(ids_caja, spacing), origen)
    log(f"{tag} distancias predichas ({len(rows)} fragmentos) en {time.perf_counter() - t:.1f} s")

    # Controles de calidad (calidad_entrada.py): espaciado de la cabecera y cortes con metal. No cambian la
    # predicción; avisan y marcan los fragmentos que caen en cortes con artefactos de metal.
    avisos = revisar_espaciado(spacing_xyz)
    metal = revisar_metal(image, spacing_xyz)
    marcados = marcar_fragmentos_con_metal(rows, metal["cortes"] if metal["avisar"] else [])
    if (aviso := aviso_metal(metal, marcados)) is not None:
        if reduccion and reduccion["cortes"]:
            aviso["texto"] += (" Se aplicó la reducción de artefactos de metal (experimental): las rayas se atenúan, "
                               "pero pueden quedar restos.")
        avisos.append(aviso)
    for aviso in avisos:
        log(f"{tag} AVISO · {aviso['titulo']}")

    gt_rows = comparison = None
    if gt_ids is not None:
        gt_caja = gt_ids[caja]
        gt_rows = _desplazar(fragment_table(gt_caja, spacing), origen)
        comparison = compare_tables(rows, gt_rows, ids_caja, gt_caja)
        log(f"{tag} ground truth: {len(gt_rows)} fragmentos, {len(comparison)} secundarios")

    arrays = {"images": images, "semantic": pred["semantic"], "ids": ids, "hu": hu_vista}
    if gt_ids is not None:
        arrays["gt_ids"] = gt_ids
    np.savez_compressed(out_dir / "volume.npz", **arrays)

    desplazamiento_mm = (origen * np.array([spacing[2], spacing[1], spacing[0]])).astype(np.float32)
    mesh_arrays = {}
    for prefix, volume in (("p", ids_caja), ("g", gt_caja if gt_ids is not None else None)):
        if volume is None:
            continue
        for fragment_id, (verts, faces) in fragment_meshes(volume, spacing).items():
            mesh_arrays[f"{prefix}_{fragment_id}_v"] = verts + desplazamiento_mm
            mesh_arrays[f"{prefix}_{fragment_id}_f"] = faces
    np.savez_compressed(out_dir / "meshes.npz", **mesh_arrays)
    log(f"{tag} mallas: {sum(len(v) for k, v in mesh_arrays.items() if k.endswith('_f')):,} triángulos")

    if raw_view:
        import gc
        gc.collect()                       # suelta lo intermedio antes de la parte que más memoria pide
        try:
            _vista_cruda(image, spacing_xyz, out_dir, log, tag, max_voxeles_vista)
        except ValueError as error:          # p. ej. sin hueso por encima del umbral: solo falta el V1
            log(f"{tag} sin vista cruda (visualizador 1): {error}")

    bone_z = np.flatnonzero((pred["semantic"] > 0).any(axis=(1, 2)))
    model_info = {k: v for k, v in info.items() if k != "device"}
    meta = {"case_id": nombre, "model": model_info, "size": size, "n_slices": int(len(images)),
            "spacing_cache_xyz": list(spacing), "spacing_native_xyz": [float(s) for s in spacing_xyz],
            "orig_shape_zyx": [int(s) for s in image.shape], "tta": bool(info.get("tta")),
            "device": str(info.get("device")), "seconds_model": pred["seconds"],
            "ms_per_slice": 1000 * pred["seconds"] / len(images), "min_voxels": min_voxels,
            "distance_method": method, "canonical": bool(info.get("canonical")), "flipped": flipped,
            "enderezado": giro, "umbrales_postproceso": umbrales, "pasos": pasos.textos,
            "avisos": avisos, "metal": metal, "reduccion_metal": reduccion, "hu_rango": [int(hu_vista.min()), int(hu_vista.max())],
            "bone_z_range": [int(bone_z[0]), int(bone_z[-1])] if len(bone_z) else None,
            "fragments_pred": rows, "fragments_gt": gt_rows, "comparison": comparison,
            "slices": [{"z": z, "classification": [round(float(p), 4) for p in pred["classification"][z]],
                        "boxes_raw": pred["boxes_raw"][z], "boxes_refined": pred["boxes_refined"][z]}
                       for z in range(len(images))]}
    meta.update(extra_meta or {})
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"{tag} listo en {time.perf_counter() - started:.0f} s → {out_dir}")
    return out_dir


def preparar_caso(ruta_imagen: str | Path, ruta_etiqueta: str | Path | None, case_id: str, canonical: bool,
                  log=print) -> tuple:
    """Lee un caso de PENGWIN, registra el paso «crudo», lo endereza si la cabecera lo pide y registra
    el paso «orientado». Devuelve (imagen, etiqueta o None, spacing_xyz, flipped, pasos).

    Sirve igual con los archivos de ``data/raw`` que con los descargados de S3 (comprimidos).
    """
    import SimpleITK as sitk
    from orientacion import flip_inplane, header_is_rotated, is_rotated
    from pasos_preproceso import Pasos, texto_matriz

    imagen_itk = sitk.ReadImage(str(ruta_imagen))
    image = sitk.GetArrayFromImage(imagen_itk)
    spacing = imagen_itk.GetSpacing()
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(ruta_etiqueta))) if ruta_etiqueta else None
    matriz = texto_matriz(imagen_itk.GetDirection())
    from_header = header_is_rotated(ruta_imagen)
    girado = is_rotated(case_id) if from_header is None else from_header
    pasos = Pasos()
    pasos.agregar_hu("crudo", "Crudo, como viene en el archivo", f"Matriz de dirección de la cabecera: {matriz}. " + (
        "Dice que el volumen está guardado girado 180° en el plano (orientación LPI): sin corregir, el sacro "
        "queda arriba y los lados al revés." if girado else
        "Es la identidad: el volumen ya viene en la orientación estándar."), image, spacing, matriz=matriz)
    flipped = False
    if canonical:
        flipped = girado
        if flipped:
            image = flip_inplane(image)
            if label is not None:
                label = flip_inplane(label)
            log(f"[{case_id}] guardado girado 180° (cabecera LPI): enderezado a la orientación canónica")
        pasos.agregar_hu("orientado", "Orientado con los metadatos", (
            "El metadato dice que viene girado 180°: se giró (filas y columnas invertidas). Ahora el sacro "
            "queda abajo, como en el resto de los casos." if flipped else
            "El metadato no indica ningún giro: no se cambió nada."), image, spacing)
    else:
        pasos.agregar_hu("orientado", "Orientado con los metadatos", "Este modelo se entrenó sin orientación "
                         "canónica: no se aplica.", image, spacing)
    return image, label, spacing, flipped, pasos


def export_case(case_id: str, checkpoint: str, out_root: str = "outputs/demo", size: int = 384,
                tta: bool = True, batch_size: int = 8, min_voxels: int = 30, raw_view: bool = True,
                reuse: bool = False, progress=None, enderezar: bool = True,
                umbral_mm: bool = True) -> Path:
    """Corre el pipeline sobre un caso de PENGWIN y escribe la carpeta del dashboard. Devuelve su ruta.

    Con ``reuse=True`` y una exportación previa del mismo caso, no vuelve a correr el modelo: lee
    ``predictions.npz`` y rehace solo el post-proceso, las distancias y las mallas. Es lo que se usa
    cuando cambia el post-proceso 3D.
    """
    from pengwin_io import find_image_path, find_label_path

    case_id = str(case_id).zfill(3)
    log = progress or (lambda message: print(message, flush=True))
    canonical = checkpoint_is_canonical(checkpoint)
    image, label, spacing_xyz, flipped, pasos = preparar_caso(find_image_path(case_id), find_label_path(case_id),
                                                               case_id, canonical, log)

    out_dir = Path(out_root) / Path(checkpoint).stem / case_id
    saved = out_dir / "predictions.npz"
    old = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else None
    pred_guardada = info = predecir = None
    if reuse and saved.exists() and old is not None:
        # Las predicciones solo sirven si la entrada al modelo es la misma: el enderezado del giro
        # axial es determinista, así que basta con que la exportación anterior lo haya hecho igual.
        if bool(old.get("enderezado")) == bool(enderezar):
            with np.load(saved) as arrays:
                pred_guardada = {"semantic": arrays["semantic"], "primary_secondary": arrays["primary_secondary"],
                                 "classification": np.array([s["classification"] for s in old["slices"]],
                                                            dtype=np.float32),
                                 "boxes_raw": [s["boxes_raw"] for s in old["slices"]],
                                 "boxes_refined": [s["boxes_refined"] for s in old["slices"]],
                                 "seconds": old["seconds_model"]}
            info = dict(old["model"], device=old["device"], tta=old["tta"], canonical=old.get("canonical"))
        else:
            log(f"[{case_id}] el enderezado cambió la entrada: no se reutilizan las predicciones")
    if pred_guardada is None:
        info, predecir = predictor_local(checkpoint, tta=tta, batch_size=batch_size, log=log)
        assert info["canonical"] == canonical, "el checkpoint y el modelo cargado no coinciden en orientación"
    return exportar_volumen(image, spacing_xyz, out_dir, info, predecir, label=label,
                            nombre=case_id, size=size, min_voxels=min_voxels, raw_view=raw_view,
                            enderezar=enderezar, umbral_mm=umbral_mm, flipped=flipped,
                            pred_guardada=pred_guardada, pasos=pasos, log=log)


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
    parser.add_argument("--umbral-voxeles", action="store_true",
                        help="Umbrales del post-proceso 3D en vóxeles (30 y 200) en vez de mm³.")
    parser.add_argument("--sin-enderezar", action="store_true",
                        help="No corrige el giro axial del paciente (normalizacion.normalizar).")
    args = parser.parse_args()
    for case_id in args.case:
        export_case(case_id, args.checkpoint, args.out_root, args.size, args.tta, args.batch_size,
                    args.min_voxels, raw_view=not args.no_raw_view, reuse=args.reuse,
                    enderezar=not args.sin_enderezar, umbral_mm=not args.umbral_voxeles)


if __name__ == "__main__":
    main()
