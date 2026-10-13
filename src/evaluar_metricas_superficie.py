"""Evalúa ASSD y HD95 3D por fragmento usando un checkpoint ya entrenado.

No optimiza pesos ni selecciona checkpoints. Ejecuta inferencia volumétrica,
construye las instancias 3D y empareja cada fragmento real con una predicción
de su misma macro-región mediante máximo IoU. Las distancias se calculan en
milímetros con el spacing físico de cada caso.
"""

from __future__ import annotations

import argparse
import csv
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
from volume_postprocess import (cache_spacing_xyz, fragment_surface_table,
                                postprocess_3d)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {}
    return {
        "media": float(array.mean()),
        "mediana": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximo": float(array.max()),
    }


def summarize(rows: list[dict]) -> dict:
    """Resume por fragmento y conserva la cobertura de los no detectados."""
    detected = [row for row in rows if row["detectado"]]
    result: dict = {
        "n_fragmentos_gt": len(rows),
        "n_fragmentos_detectados": len(detected),
        "n_fragmentos_no_detectados": len(rows) - len(detected),
        "cobertura_fragmentos": float(len(detected) / len(rows)) if rows else 0.0,
        "assd_mm": _stats([float(row["assd_mm"]) for row in detected]),
        "hd95_mm": _stats([float(row["hd95_mm"]) for row in detected]),
    }
    result["por_tipo"] = {}
    for kind in ("principal", "secundario"):
        group = [row for row in rows if row["tipo"] == kind]
        group_detected = [row for row in group if row["detectado"]]
        result["por_tipo"][kind] = {
            "n_gt": len(group),
            "n_detectados": len(group_detected),
            "cobertura": float(len(group_detected) / len(group)) if group else 0.0,
            "assd_mm": _stats([float(row["assd_mm"]) for row in group_detected]),
            "hd95_mm": _stats([float(row["hd95_mm"]) for row in group_detected]),
        }
    result["por_region"] = {}
    for region in ("sacro", "coxal_izquierdo", "coxal_derecho"):
        group = [row for row in rows if row["region"] == region]
        group_detected = [row for row in group if row["detectado"]]
        result["por_region"][region] = {
            "n_gt": len(group),
            "n_detectados": len(group_detected),
            "cobertura": float(len(group_detected) / len(group)) if group else 0.0,
            "assd_mm": _stats([float(row["assd_mm"]) for row in group_detected]),
            "hd95_mm": _stats([float(row["hd95_mm"]) for row in group_detected]),
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "case_id", "region", "tipo", "gt_id", "pred_id",
              "volume_voxels", "dice", "iou", "detectado", "assd_mm", "hd95_mm"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--two-pass", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--min-voxels", type=int, default=30)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    cache_dir = Path(args.cache_dir).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_two_pass = load_model([str(checkpoint)], device, args.tta)
    two_pass = bool(model_two_pass or args.two_pass)
    size = int(cache_dir.name.split("_")[1])
    index = read_index(cache_dir)
    case_ids = ([str(case).zfill(3) for case in args.cases]
                if args.cases else cases_of_split(cache_dir, args.split))

    print(f"{len(case_ids)} casos de {args.split} | {device} | TTA={args.tta} | "
          f"checkpoint={checkpoint.name}", flush=True)
    started = time.time()
    all_rows: list[dict] = []
    per_case: list[dict] = []
    for number, case_id in enumerate(case_ids, 1):
        case_started = time.time()
        case = load_case_cache(cache_dir, case_id, canonical=getattr(model, "canonical", False))
        _, predictions = evaluate_case(
            model, case_id, case["images"], case["labels"], device,
            batch_size=args.batch_size, confidence=args.confidence,
            two_pass=two_pass, keep_predictions=True)
        if predictions is None or "primary_secondary" not in predictions:
            raise RuntimeError("HD95/ASSD requiere un modelo con salida primary_secondary.")
        predicted_volume = postprocess_3d(
            predictions["semantic"], predictions["primary_secondary"], args.min_voxels)
        spacing = cache_spacing_xyz(index[case_id], size)
        rows = fragment_surface_table(predicted_volume, case["labels"], spacing)
        for row in rows:
            row["split"] = args.split
            row["case_id"] = case_id
        case_summary = summarize(rows)
        case_summary.update({
            "case_id": case_id,
            "spacing_xyz_mm": spacing,
            "segundos": round(time.time() - case_started, 1),
        })
        per_case.append(case_summary)
        all_rows.extend(rows)
        print(f"  {number}/{len(case_ids)} caso {case_id}: "
              f"fragmentos={case_summary['n_fragmentos_gt']} "
              f"cobertura={case_summary['cobertura_fragmentos']:.1%} "
              f"ASSD={case_summary['assd_mm'].get('media', float('nan')):.2f} mm "
              f"HD95={case_summary['hd95_mm'].get('media', float('nan')):.2f} mm",
              flush=True)

    result = {
        "metodo": "superficies_3d_por_fragmento_emparejado_por_iou",
        "checkpoint": str(checkpoint),
        "split": args.split,
        "cases": case_ids,
        "tta": bool(args.tta),
        "two_pass": two_pass,
        "confidence": float(args.confidence),
        "min_voxels": int(args.min_voxels),
        "device": str(device),
        "seconds": round(time.time() - started, 1),
        "definiciones": {
            "ASSD": "Promedio simétrico de las distancias entre ambas superficies, en mm.",
            "HD95": "Percentil 95 de las distancias bidireccionales de superficie, en mm.",
            "matching": "Uno a uno por máximo IoU dentro de la misma macro-región.",
            "no_detectado": "ASSD y HD95 quedan null y se reportan mediante cobertura.",
        },
        "resumen": summarize(all_rows),
        "por_caso": per_case,
        "por_fragmento": all_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_output = Path(args.csv_output) if args.csv_output else output.with_suffix(".csv")
    write_csv(csv_output, all_rows)

    summary = result["resumen"]
    print("\n=== Métricas opcionales de superficie por fragmento ===")
    print(f"Fragmentos GT:          {summary['n_fragmentos_gt']}")
    print(f"Fragmentos detectados:  {summary['n_fragmentos_detectados']}")
    print(f"Cobertura:              {summary['cobertura_fragmentos']:.2%}")
    print(f"ASSD media:             {summary['assd_mm'].get('media', float('nan')):.3f} mm")
    print(f"ASSD mediana:           {summary['assd_mm'].get('mediana', float('nan')):.3f} mm")
    print(f"HD95 media:             {summary['hd95_mm'].get('media', float('nan')):.3f} mm")
    print(f"HD95 mediana:           {summary['hd95_mm'].get('mediana', float('nan')):.3f} mm")
    print("Menor es mejor. Las medias incluyen solo parejas detectadas; la cobertura declara las omisiones.")
    print("JSON:", output)
    print("CSV: ", csv_output)


if __name__ == "__main__":
    main()
