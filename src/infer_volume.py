"""Inferencia sobre el volumen completo con post-proceso 3D y distancias en mm (Fase 6).

Por caso: predice todos los cortes (``evaluate_case``), arma el volumen de IDs con
``volume_postprocess.postprocess_3d`` y compara contra el ground truth:

- Dice e IoU 3D por fragmento (principal y secundarios);
- distancia en mm de cada fragmento secundario al principal de su región, predicha y real,
  con su error absoluto; opcionalmente ASSD y HD95 contra el fragmento real.

El spacing sale de ``index.csv`` ajustado a la resolución del caché estirado. Uso:

    python src/infer_volume.py --checkpoint outputs/s10/c2_w32_cbam1_sec3_e60_s42.pth \\
        --cache-dir data/cache/s10/cache_384 --split val --tta --surface-metrics \\
        --output outputs/s10/c2_volumen_val.json
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
from volume_postprocess import (cache_spacing_xyz, fragment_distance_table, postprocess_3d,
                                summarize_volume, umbrales_en_voxeles)


def read_index(cache_dir: Path) -> dict[str, dict]:
    with open(cache_dir.parent / "index.csv", newline="") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


def aggregate(cases: list[dict]) -> dict:
    out: dict = {"n_cases": len(cases)}
    for key in ("dice_principal", "dice_secundario", "iou_principal", "iou_secundario",
                "dice_todos", "iou_todos", "perdidos_secundario"):
        values = [c["volumen"][key] for c in cases if key in c["volumen"]]
        if values:
            out[key] = float(np.mean(values))
    rows = [r for c in cases for r in c["distancias"]]
    out["n_fragmentos_secundarios"] = len(rows)
    errors = [r["absolute_error_mm"] for r in rows if r["absolute_error_mm"] is not None]
    out["n_con_distancia_predicha"] = len(errors)
    if errors:
        out["error_distancia_mm_media"] = float(np.mean(errors))
        out["error_distancia_mm_mediana"] = float(np.median(errors))
        out["error_distancia_mm_p90"] = float(np.percentile(errors, 90))
    for key in ("assd_mm", "hd95_mm"):
        values = [r[key] for r in rows if key in r]
        if values:
            out[f"{key}_media"] = float(np.mean(values))
    gt = [r["ground_truth_mm"] for r in rows if r["predicted_mm"] is not None]
    pred = [r["predicted_mm"] for r in rows if r["predicted_mm"] is not None]
    if len(gt) > 2 and np.std(gt) > 0 and np.std(pred) > 0:
        out["correlacion_distancia"] = float(np.corrcoef(gt, pred)[0, 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, nargs="+")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--two-pass", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-voxels", type=int, default=30)
    parser.add_argument("--distance-nofill", action="store_true",
                        help="Mide las distancias sobre el volumen sin rellenar huecos entre fragmentos.")
    parser.add_argument("--surface-metrics", action="store_true", help="Agrega ASSD y HD95.")
    parser.add_argument("--umbral-mm", action="store_true",
                        help="Umbrales del post-proceso 3D en mm³ (volume_postprocess.umbrales_en_voxeles), "
                             "no en vóxeles: no dependen del grosor de los cortes.")
    parser.add_argument("--zstep", type=int, default=1,
                        help="Conserva 1 de cada k cortes (imagen y etiqueta) y multiplica el espaciado z "
                             "por k: simula un volumen con cortes más gruesos. Solo validación.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.split == "test":
        print("AVISO: test se evalúa una sola vez, con la configuración ya congelada en validación.")
    if args.zstep > 1 and args.split == "test":
        raise SystemExit("--zstep es una prueba de estrés: solo validación.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, two_pass = load_model(args.checkpoint, device, args.tta)
    two_pass = two_pass or args.two_pass
    cache_dir = Path(args.cache_dir)
    size = int(cache_dir.name.split("_")[1])
    index = read_index(cache_dir)
    case_ids = [str(c).zfill(3) for c in args.cases] if args.cases else cases_of_split(cache_dir, args.split)

    cases = []
    for number, case_id in enumerate(case_ids, 1):
        case = load_case_cache(cache_dir, case_id, canonical=getattr(model, "canonical", False))
        case["images"], case["labels"] = case["images"][::args.zstep], case["labels"][::args.zstep]
        spacing = cache_spacing_xyz(index[case_id], size)
        spacing = (spacing[0], spacing[1], spacing[2] * args.zstep)
        min_voxels, min_region = (umbrales_en_voxeles(spacing) if args.umbral_mm
                                  else (args.min_voxels, 200))
        started = time.time()
        _, pred = evaluate_case(model, case_id, case["images"], case["labels"], device,
                                args.batch_size, two_pass=two_pass, keep_predictions=True)
        if "primary_secondary" not in pred:
            raise SystemExit("El post-proceso 3D necesita un modelo con instancia 'primary'.")
        t_model = time.time() - started
        volume = postprocess_3d(pred["semantic"], pred["primary_secondary"], min_voxels, min_region)
        t_total = time.time() - started
        dist_volume = (postprocess_3d(pred["semantic"], pred["primary_secondary"], min_voxels, min_region,
                                      fill_unlabeled=False) if args.distance_nofill else volume)
        distances = fragment_distance_table(dist_volume, case["labels"], spacing, args.surface_metrics)
        cases.append({"case_id": case_id, "n_slices": int(len(volume)),
                      "spacing_xyz_mm": spacing, "min_voxels": min_voxels, "min_region_voxels": min_region,
                      "segundos_modelo": round(t_model, 1),
                      "segundos_total": round(t_total, 1),
                      "volumen": summarize_volume(volume, case["labels"]), "distancias": distances})
        v = cases[-1]["volumen"]
        print(f"  {number}/{len(case_ids)} caso {case_id}: dice3D principal={v.get('dice_principal', 0):.3f} "
              f"secundario={v.get('dice_secundario', float('nan')):.3f} | "
              f"{len(distances)} fragmentos secundarios | {t_total:.0f} s", flush=True)

    result = {"checkpoint": args.checkpoint, "split": args.split, "tta": args.tta,
              "two_pass": two_pass, "min_voxels": args.min_voxels, "umbral_mm": args.umbral_mm,
              "zstep": args.zstep, "device": str(device),
              "resumen": aggregate(cases), "casos": cases}
    for key, value in result["resumen"].items():
        print(f"  {key:32s} {value:.3f}" if isinstance(value, float) else f"  {key:32s} {value}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
