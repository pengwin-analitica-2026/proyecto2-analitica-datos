"""Diagnóstico del Dice de instancias sobre VALIDACIÓN (nunca test).

Responde tres preguntas con una sola pasada del modelo:

1. ¿Dónde se pierde el Dice? Tabla por tipo (primario/secundario), tamaño del fragmento,
   posición del corte, región y número de fragmentos reales en la región.
2. ¿Cuál es el techo de cada pieza? Oráculos: partición real con nuestra máscara de hueso,
   máscara real con nuestra partición y el techo de la propia formulación
   (mapa primario/secundario real + componentes conectados).
3. ¿Qué post-proceso ayuda sin reentrenar? Variantes de ``primary_secondary_to_instances``,
   post-proceso 3D y corte por la cabeza de borde.

La métrica replica ``evaluate_volumes.matched_fragments`` (un valor por fragmento real y corte,
0 si queda sin pareja) con una implementación vectorizada equivalente. Además cuenta los
fragmentos predichos sobrantes, que la métrica del curso no penaliza.

Uso:
    python src/analisis_instancias.py --checkpoint outputs/s10/d5_ema_w32_e60_s42.pth \
        --cache-dir data/cache/s10/cache_384 --tta --output-dir outputs/s10/analisis_d5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_cache import from_uint8, load_case_cache  # noqa: E402
from evaluate_volumes import (PRIMARY_IDS, REGION_RANGES, cases_of_split, load_model,  # noqa: E402
                              relative_positions, semantic_from_labels)
from instance_postprocess import primary_secondary_to_instances, semantic_to_instances  # noqa: E402
from volume_postprocess import postprocess_3d  # noqa: E402

STARTS = (1, 11, 21)


def match_slice(pred: np.ndarray, truth: np.ndarray) -> tuple[list[tuple[int, float, int]], int]:
    """Igual que ``matched_fragments`` pero con una tabla de contingencia.

    Devuelve [(id real, Dice, área real)] y el número de fragmentos predichos sin pareja.
    """
    joint = np.bincount(truth.ravel().astype(np.int64) * 32 + pred.ravel().astype(np.int64),
                        minlength=32 * 32).reshape(32, 32)
    gt_area, pred_area = joint.sum(1), joint.sum(0)
    rows, extra = [], 0
    for low, high in REGION_RANGES:
        gt_ids = [i for i in range(low, high + 1) if gt_area[i] > 0]
        pred_ids = [i for i in range(low, high + 1) if pred_area[i] > 0]
        used: set[int] = set()
        for g in gt_ids:
            best, best_iou = None, 0.0
            for p in pred_ids:
                if p in used:
                    continue
                inter = joint[g, p]
                iou = inter / max(gt_area[g] + pred_area[p] - inter, 1)
                if iou > best_iou:
                    best, best_iou = p, iou
            if best is None:
                rows.append((g, 0.0, int(gt_area[g])))
                continue
            used.add(best)
            rows.append((g, float(2 * joint[g, best] / max(gt_area[g] + pred_area[best], 1)),
                         int(gt_area[g])))
        extra += len(pred_ids) - len(used)
    return rows, extra


def ps_from_labels(labels: np.ndarray) -> np.ndarray:
    """Mapa real de 7 clases: primario/secundario por región."""
    out = np.zeros(labels.shape, dtype=np.uint8)
    for region, (low, high) in enumerate(REGION_RANGES):
        out[labels == low] = 2 * region + 1
        out[(labels > low) & (labels <= high)] = 2 * region + 2
    return out


def nearest_fill(target_mask: np.ndarray, source: np.ndarray, region_range: tuple[int, int]) -> np.ndarray:
    """Para los píxeles de ``target_mask`` devuelve la etiqueta más cercana de ``source`` en el rango."""
    low, high = region_range
    valid = (source >= low) & (source <= high)
    out = np.zeros(source.shape, dtype=np.uint8)
    if not valid.any() or not target_mask.any():
        return out
    _, idx = ndi.distance_transform_edt(~valid, return_indices=True)
    out[target_mask] = source[idx[0][target_mask], idx[1][target_mask]]
    return out


def oracle_partition(semantic: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nuestra máscara de hueso con la partición real (etiqueta real más cercana)."""
    out = np.zeros_like(labels)
    for z in range(len(labels)):
        for region, rng in enumerate(REGION_RANGES):
            mask = semantic[z] == region + 1
            if mask.any():
                filled = nearest_fill(mask, labels[z], rng)
                filled[mask & (filled == 0)] = rng[0]   # región sin GT en el corte
                out[z][mask] = filled[mask]
    return out


def oracle_mask(pred_instances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """La máscara real con nuestra partición (fragmento predicho más cercano)."""
    out = np.zeros_like(labels)
    gt_sem = semantic_from_labels(labels)
    for z in range(len(labels)):
        for region, rng in enumerate(REGION_RANGES):
            mask = gt_sem[z] == region + 1
            if mask.any():
                out[z][mask] = nearest_fill(mask, pred_instances[z], rng)[mask]
    return out


def boundary_cut(ps: np.ndarray, semantic: np.ndarray, boundary: np.ndarray, threshold: float) -> np.ndarray:
    """Corta por el borde predicho antes de etiquetar y devuelve los píxeles al vecino más cercano."""
    cut = ps.copy()
    cut[boundary >= threshold] = 0
    inst = primary_secondary_to_instances(cut, semantic)
    if inst.ndim == 2:
        inst = inst[None]
    for z in range(len(inst)):
        for region, rng in enumerate(REGION_RANGES):
            lost = (semantic[z] == region + 1) & (ps[z] > 0) & (inst[z] == 0) & (boundary[z] >= threshold)
            if lost.any():
                inst[z][lost] = nearest_fill(lost, inst[z], rng)[lost]
    return inst


def as3d(x: np.ndarray) -> np.ndarray:
    return x[None] if x.ndim == 2 else x


def build_variants(sem: np.ndarray, ps: np.ndarray, boundary: np.ndarray | None,
                   labels: np.ndarray) -> dict[str, np.ndarray]:
    gt_sem, gt_ps = semantic_from_labels(labels), ps_from_labels(labels)
    base = as3d(primary_secondary_to_instances(ps, sem))
    variants = {
        "base (min 12)": base,
        "min_size 1": as3d(primary_secondary_to_instances(ps, sem, min_size=1)),
        "min_size 4": as3d(primary_secondary_to_instances(ps, sem, min_size=4)),
        "min_size 30": as3d(primary_secondary_to_instances(ps, sem, min_size=30)),
        "rellenar huecos": as3d(primary_secondary_to_instances(ps, sem, fill_holes=True)),
        "absorber <20": as3d(primary_secondary_to_instances(ps, sem, absorb_small=20)),
        "absorber <50": as3d(primary_secondary_to_instances(ps, sem, absorb_small=50)),
        "min 1 + absorber <20": as3d(primary_secondary_to_instances(ps, sem, min_size=1, absorb_small=20)),
        "sin restringir a semantica": as3d(primary_secondary_to_instances(ps, None)),
        "3D (con relleno)": postprocess_3d(sem, ps),
        "3D (sin relleno)": postprocess_3d(sem, ps, fill_unlabeled=False),
        "3D min_voxels 1": postprocess_3d(sem, ps, min_voxels=1),
        "solo semantica (componentes)": as3d(semantic_to_instances(sem)),
        "un fragmento por region": np.where(sem > 0, np.asarray([0, *STARTS], dtype=np.uint8)[sem], 0).astype(np.uint8),
        "ORACULO particion real + nuestra mascara": oracle_partition(sem, labels),
        "ORACULO mascara real + nuestra particion": oracle_mask(base, labels),
        "ORACULO formulacion 2D (mapa P/S real)": as3d(primary_secondary_to_instances(gt_ps, gt_sem, min_size=1)),
        "ORACULO formulacion 3D (mapa P/S real)": postprocess_3d(gt_sem, gt_ps, min_voxels=1, min_region_voxels=1),
        "ORACULO P/S real + nuestra mascara": as3d(primary_secondary_to_instances(
            np.where(sem == gt_sem, gt_ps, np.where(sem > 0, 2 * (sem.astype(np.int16) - 1) + 1, 0)).astype(np.uint8),
            sem, min_size=1)),
    }
    if boundary is not None:
        for threshold in (0.3, 0.5, 0.7):
            variants[f"corte por borde >= {threshold}"] = boundary_cut(ps, sem, boundary, threshold)
    return variants


def area_bin(area: int) -> str:
    for limit, name in ((25, "1) <25 px"), (100, "2) 25-99"), (400, "3) 100-399"), (1600, "4) 400-1599")):
        if area < limit:
            return name
    return "5) >=1600"


def rel_bin(rel: float) -> str:
    if not np.isfinite(rel):
        return "sin dato"
    if rel < 0.15:
        return "1) 0-15% (inicio)"
    if rel > 0.85:
        return "3) 85-100% (final)"
    return "2) 15-85% (centro)"


def table(rows: list[dict], key) -> list[dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row["dice"])
    total_loss = sum(1 - r["dice"] for r in rows) or 1.0
    out = []
    for name in sorted(groups):
        values = np.asarray(groups[name])
        out.append({"grupo": name, "n": int(values.size), "frac_n": round(values.size / len(rows), 3),
                    "dice": round(float(values.mean()), 3),
                    "sin_pareja": round(float((values == 0).mean()), 3),
                    "frac_perdida": round(float((1 - values).sum() / total_loss), 3)})
    return out


def save_example(path: Path, image: np.ndarray, truth: np.ndarray, pred: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ys, xs = np.where((truth > 0) | (pred > 0))
    y0, y1 = max(ys.min() - 12, 0), min(ys.max() + 12, image.shape[0])
    x0, x1 = max(xs.min() - 12, 0), min(xs.max() + 12, image.shape[1])
    crop = (slice(y0, y1), slice(x0, x1))
    cmap = plt.get_cmap("tab20", 31)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, data, name in zip(axes, (None, truth, pred), ("CT", "Real", "Prediccion")):
        ax.imshow(image[crop], cmap="gray")
        if data is not None:
            ax.imshow(np.ma.masked_equal(data[crop], 0), cmap=cmap, vmin=0, vmax=30, alpha=0.6, interpolation="nearest")
            ids = [int(v) for v in np.unique(data[crop]) if v > 0]
            name = f"{name} ids={ids}"
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


@torch.inference_mode()
def predict_case(model, images_u8: np.ndarray, device, batch_size: int = 16):
    sems, pss, bounds = [], [], []
    for start in range(0, len(images_u8), batch_size):
        x = torch.from_numpy(from_uint8(images_u8[start:start + batch_size]))[:, None].to(device)
        out = model(x)
        sems.append(out["semantic"].argmax(1).cpu().numpy().astype(np.uint8))
        pss.append(out["instance"].argmax(1).cpu().numpy().astype(np.uint8))
        if "boundary" in out:
            bounds.append(torch.sigmoid(out["boundary"].squeeze(1)).cpu().numpy().astype(np.float32))
    return np.concatenate(sems), np.concatenate(pss), (np.concatenate(bounds) if bounds else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="Solo validación o entrenamiento: este script no se corre sobre test.")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--examples-per-case", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device, args.tta)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    (out_dir / "ejemplos").mkdir(parents=True, exist_ok=True)

    base_rows: list[dict] = []
    scores: dict[str, dict[str, list]] = defaultdict(lambda: {"all": [], "pri": [], "sec": [], "extra": 0})
    per_case: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    cases = args.cases or cases_of_split(cache_dir, args.split)
    for number, case_id in enumerate(cases, 1):
        case = load_case_cache(cache_dir, case_id, canonical=getattr(model, "canonical", False))
        images, labels = case["images"], case["labels"]
        sem, ps, boundary = predict_case(model, images, device)
        has_bone = (labels != 0).any(axis=(1, 2))
        rel = relative_positions(has_bone)
        variants = build_variants(sem, ps, boundary, labels)
        n_gt_region = {(z, r): len([v for v in np.unique(labels[z]) if low <= v <= high])
                       for z in np.flatnonzero(has_bone) for r, (low, high) in enumerate(REGION_RANGES)}
        for name, inst in variants.items():
            for z in np.flatnonzero(has_bone):
                rows, extra = match_slice(inst[z], labels[z])
                bucket = scores[name]
                bucket["extra"] += extra
                for gt_id, dice, area in rows:
                    bucket["all"].append(dice)
                    bucket["pri" if gt_id in PRIMARY_IDS else "sec"].append(dice)
                    per_case[name][case_id].append(dice)
                    if name == "base (min 12)":
                        region = (gt_id - 1) // 10
                        base_rows.append({"case": case_id, "z": int(z), "rel": float(rel[z]), "id": gt_id,
                                          "region": region, "area": area, "dice": dice,
                                          "primary": gt_id in PRIMARY_IDS,
                                          "n_gt_region": n_gt_region[(int(z), region)]})
        # Ejemplos del caso: los peores fragmentos grandes (lo que más pesa y se puede ver).
        mine = [r for r in base_rows if r["case"] == case_id]
        worst = sorted([r for r in mine if not r["primary"] and r["area"] >= 150], key=lambda r: r["dice"])
        worst += sorted([r for r in mine if r["primary"] and r["area"] >= 600], key=lambda r: r["dice"])[:1]
        seen = set()
        for row in worst:
            if row["z"] in seen or len(seen) >= args.examples_per_case + 1:
                continue
            seen.add(row["z"])
            z = row["z"]
            save_example(out_dir / "ejemplos" / f"{case_id}_z{z:03d}_id{row['id']}.png", images[z], labels[z],
                         variants["base (min 12)"][z],
                         f"caso {case_id} corte {z} | fragmento real {row['id']} area {row['area']} px Dice {row['dice']:.2f}")
        print(f"{number}/{len(cases)} caso {case_id}: base "
              f"{np.mean(per_case['base (min 12)'][case_id]):.3f}", flush=True)

    summary = {"checkpoints": args.checkpoint, "tta": args.tta, "split": args.split, "n_fragmentos": len(base_rows)}
    summary["variantes"] = [
        {"variante": name, "dice_todos": round(float(np.mean(b["all"])), 4),
         "dice_primario": round(float(np.mean(b["pri"])), 4),
         "dice_secundario": round(float(np.mean(b["sec"])), 4) if b["sec"] else None,
         "sin_pareja_sec": round(float(np.mean(np.asarray(b["sec"]) == 0)), 4) if b["sec"] else None,
         "fragmentos_predichos_sobrantes": int(b["extra"]),
         "media_por_caso": round(float(np.mean([np.mean(v) for v in per_case[name].values()])), 4)}
        for name, b in scores.items()]
    summary["por_caso_base"] = {c: round(float(np.mean(v)), 4) for c, v in per_case["base (min 12)"].items()}
    summary["tablas_base"] = {
        "tipo": table(base_rows, lambda r: "primario" if r["primary"] else "secundario"),
        "area": table(base_rows, lambda r: area_bin(r["area"])),
        "area_secundarios": table([r for r in base_rows if not r["primary"]], lambda r: area_bin(r["area"])),
        "area_primarios": table([r for r in base_rows if r["primary"]], lambda r: area_bin(r["area"])),
        "posicion": table(base_rows, lambda r: rel_bin(r["rel"])),
        "region": table(base_rows, lambda r: ("sacro", "coxal_izq", "coxal_der")[r["region"]]),
        "fragmentos_reales_en_la_region": table(base_rows, lambda r: f"{min(r['n_gt_region'], 4)}{'+' if r['n_gt_region'] >= 4 else ''}"),
        "primario_segun_fragmentos_en_region": table([r for r in base_rows if r["primary"]],
                                                     lambda r: "solo" if r["n_gt_region"] == 1 else "con secundarios"),
    }
    (out_dir / "analisis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez_compressed(out_dir / "filas_base.npz", **{k: np.asarray([r[k] for r in base_rows])
                                                        for k in ("case", "z", "rel", "id", "area", "dice")})
    print(json.dumps(summary["variantes"], indent=1, ensure_ascii=False))
    for name, rows in summary["tablas_base"].items():
        print("\n[", name, "]")
        for row in rows:
            print("  ", row)


if __name__ == "__main__":
    main()
