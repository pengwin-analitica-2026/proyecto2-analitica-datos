"""HD95 y ASSD en milímetros por fragmento óseo, sobre el volumen completo.

Por caso: predice todos los cortes, arma el volumen de IDs con ``postprocess_3d`` y empareja en 3D
cada fragmento real con el predicho de mayor IoU de su misma región (``match_fragments``; un
fragmento predicho se usa una sola vez). Para cada fragmento real:

- si tiene pareja, mide ASSD y HD95 entre las dos superficies, en mm, con el spacing de la
  cabecera ajustado a la resolución del caché (``cache_spacing_xyz``);
- si no tiene pareja (IoU 0 con todos los predichos de su región), cuenta como no detectado.

El resumen da, para todos los fragmentos, los principales y los secundarios: cuántos hay, cuántos
se detectaron (cobertura), y media, mediana y percentil 90 de HD95 y ASSD entre los detectados.
También cuenta los fragmentos predichos que no se emparejan con ninguno real.

Uso:

    python src/metricas_superficie.py --checkpoint outputs/s10/modelo.pth \\
        --cache-dir data/cache/s10/cache_384 --split val --tta --output outputs/superficie_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_cache import load_case_cache
from evaluate_volumes import cases_of_split, evaluate_case, load_model
from infer_volume import read_index
from volume_postprocess import cache_spacing_xyz, match_fragments, postprocess_3d, surface_distances

REGION_NAMES = {0: "sacro", 1: "coxal_izquierdo", 2: "coxal_derecho"}
IOU_UMBRALES = (0.1, 0.5)


def fragment_rows(prediction: np.ndarray, truth: np.ndarray,
                  spacing_xyz: tuple[float, float, float]) -> tuple[list[dict], int]:
    """Una fila por fragmento real y el número de fragmentos predichos sin pareja."""
    matches = match_fragments(prediction, truth)
    rows = []
    for m in matches:
        row = {"gt_id": m["gt_id"], "region": REGION_NAMES[(m["gt_id"] - 1) // 10],
               "principal": bool(m["primary"]), "volume_voxels": m["volume_voxels"],
               "pred_id": m["pred_id"], "iou": m["iou"], "dice": m["dice"],
               "detectado": m["pred_id"] is not None, "assd_mm": None, "hd95_mm": None}
        if m["pred_id"] is not None:
            row["assd_mm"], row["hd95_mm"] = surface_distances(
                prediction == m["pred_id"], truth == m["gt_id"], spacing_xyz)
        rows.append(row)
    used = {m["pred_id"] for m in matches if m["pred_id"] is not None}
    extra = len({int(v) for v in np.unique(prediction) if v > 0} - used)
    return rows, extra


def summarize(rows: list[dict]) -> dict:
    out: dict = {"n_fragmentos": len(rows)}
    if not rows:
        return out
    found = [r for r in rows if r["detectado"]]
    out["n_detectados"] = len(found)
    out["n_no_detectados"] = len(rows) - len(found)
    out["cobertura"] = len(found) / len(rows)
    for umbral in IOU_UMBRALES:
        out[f"cobertura_iou_{umbral}"] = float(np.mean([r["iou"] >= umbral for r in rows]))
    out["iou_media"] = float(np.mean([r["iou"] for r in rows]))
    out["dice_media"] = float(np.mean([r["dice"] for r in rows]))
    for key in ("hd95_mm", "assd_mm"):
        values = np.asarray([r[key] for r in found], dtype=np.float64)
        if len(values):
            out[f"{key}_media"] = float(values.mean())
            out[f"{key}_mediana"] = float(np.median(values))
            out[f"{key}_p90"] = float(np.percentile(values, 90))
            out[f"{key}_max"] = float(values.max())
    return out


def summarize_groups(rows: list[dict]) -> dict:
    groups = {"todos": rows, "principal": [r for r in rows if r["principal"]],
              "secundario": [r for r in rows if not r["principal"]]}
    for name in REGION_NAMES.values():
        groups[name] = [r for r in rows if r["region"] == name]
    return {name: summarize(group) for name, group in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, nargs="+")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-voxels", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.split == "test":
        print("AVISO: test se evalúa una sola vez, con la configuración ya congelada en validación.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, two_pass = load_model(args.checkpoint, device, args.tta)
    cache_dir = Path(args.cache_dir)
    size = int(cache_dir.name.split("_")[1])
    index = read_index(cache_dir)
    case_ids = [str(c).zfill(3) for c in args.cases] if args.cases else cases_of_split(cache_dir, args.split)

    cases, all_rows = [], []
    for number, case_id in enumerate(case_ids, 1):
        started = time.time()
        case = load_case_cache(cache_dir, case_id, canonical=getattr(model, "canonical", False))
        _, pred = evaluate_case(model, case_id, case["images"], case["labels"], device,
                                args.batch_size, two_pass=two_pass, keep_predictions=True)
        if "primary_secondary" not in pred:
            raise SystemExit("El post-proceso 3D necesita un modelo con instancia 'primary'.")
        volume = postprocess_3d(pred["semantic"], pred["primary_secondary"], args.min_voxels)
        spacing = cache_spacing_xyz(index[case_id], size)
        rows, extra = fragment_rows(volume, case["labels"], spacing)
        for row in rows:
            row["case_id"] = case_id
        all_rows.extend(rows)
        resumen = summarize(rows)
        cases.append({"case_id": case_id, "spacing_xyz_mm": spacing, "predichos_sin_pareja": extra,
                      "resumen": resumen, "fragmentos": rows})
        print(f"  {number}/{len(case_ids)} caso {case_id}: {resumen['n_detectados']}/{resumen['n_fragmentos']} "
              f"detectados | HD95 mediana {resumen.get('hd95_mm_mediana', float('nan')):.2f} mm | "
              f"ASSD mediana {resumen.get('assd_mm_mediana', float('nan')):.2f} mm | "
              f"{time.time() - started:.0f} s", flush=True)

    result = {"checkpoint": args.checkpoint, "split": args.split, "tta": args.tta,
              "min_voxels": args.min_voxels, "device": str(device), "n_casos": len(cases),
              "predichos_sin_pareja": int(sum(c["predichos_sin_pareja"] for c in cases)),
              "resumen": summarize_groups(all_rows), "casos": cases}
    for name, group in result["resumen"].items():
        if group.get("n_fragmentos"):
            print(f"  {name:16s} n={group['n_fragmentos']:3d} cobertura={group['cobertura']:.3f} "
                  f"HD95 media/mediana={group.get('hd95_mm_media', float('nan')):.2f}/"
                  f"{group.get('hd95_mm_mediana', float('nan')):.2f} mm "
                  f"ASSD media/mediana={group.get('assd_mm_media', float('nan')):.2f}/"
                  f"{group.get('assd_mm_mediana', float('nan')):.2f} mm")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
