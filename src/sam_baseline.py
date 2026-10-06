"""Línea base SAM zero-shot (Semana 10, objetivo 10G).

SAM ViT-B NO forma parte del sistema: solo se usa como referencia. Recibe como
prompt la caja que predice NUESTRO detector y devuelve una máscara binaria sin
clasificar fragmentos. Se compara, región por región, con la máscara semántica de
nuestra red dentro de la misma caja, contra la verdad de terreno.

Métrica por (corte, región presente en el GT): Dice e IoU de la región completa
(sacro / coxal izquierdo / coxal derecho). Si el detector no produce caja para esa
región, ambos métodos valen 0 (misma regla para los dos).

Uso:
    python src/sam_baseline.py --checkpoint outputs/s10/d5_ema_w32_e60_s42.pth \
        --sam-checkpoint /content/sam_vit_b_01ec64.pth --cache-dir data/cache/s10/cache_384 \
        --stride 3 --output outputs/s10/sam_baseline_val.json
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
from build_case_cache import from_uint8, load_case_cache  # noqa: E402
from evaluate_volumes import cases_of_split, load_model, semantic_from_labels  # noqa: E402
from inference_week10 import infer_slice, _box_to_pixels  # noqa: E402

REGIONS = {0: "sacro", 1: "coxal_izquierdo", 2: "coxal_derecho"}


def dice_iou(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    inter = float((pred & truth).sum())
    total = float(pred.sum() + truth.sum())
    union = float((pred | truth).sum())
    return (2 * inter / total if total else 1.0), (inter / union if union else 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--stride", type=int, default=3, help="un corte de cada N con hueso")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from segment_anything import SamPredictor, sam_model_registry  # pip install segment-anything
    device = torch.device(args.device)
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    model, _ = load_model([args.checkpoint], device, tta=False)

    cache_dir = Path(args.cache_dir)
    rows = {name: {"sam": [], "nuestro": []} for name in REGIONS.values()}
    sam_time, n = 0.0, 0
    for case_id in cases_of_split(cache_dir, args.split):
        case = load_case_cache(cache_dir, case_id)
        images, labels = case["images"], case["labels"]
        gt_sem = semantic_from_labels(labels)
        for z in np.flatnonzero((labels != 0).any(axis=(1, 2)))[:: args.stride]:
            tensor = torch.from_numpy(from_uint8(images[z]))[None, None].to(device)
            result = infer_slice(model, tensor)
            height, width = images[z].shape
            rgb = np.repeat(images[z][..., None], 3, axis=2)
            start = time.perf_counter()
            predictor.set_image(rgb)
            masks_sam = {}
            best = {}
            for det in result["detections"]:
                rid = int(det["region_id"])
                if rid not in best or det["score"] > best[rid]["score"]:
                    best[rid] = det
            for rid, det in best.items():
                x0, y0, x1, y1 = _box_to_pixels(det["box"], height, width)
                mask, _, _ = predictor.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)
                masks_sam[rid] = (mask[0], (x0, y0, x1, y1))
            sam_time += time.perf_counter() - start
            n += 1
            for rid, name in REGIONS.items():
                truth = gt_sem[z] == rid + 1
                if not truth.any():
                    continue
                if rid in masks_sam:
                    sam_mask, (x0, y0, x1, y1) = masks_sam[rid]
                    ours = np.zeros_like(truth)
                    ours[y0:y1, x0:x1] = (result["semantic"][y0:y1, x0:x1] == rid + 1)
                    rows[name]["sam"].append(dice_iou(sam_mask, truth))
                    rows[name]["nuestro"].append(dice_iou(ours, truth))
                else:
                    rows[name]["sam"].append((0.0, 0.0))
                    rows[name]["nuestro"].append((0.0, 0.0))
        print("caso", case_id, "cortes acumulados", n, flush=True)

    summary = {"split": args.split, "stride": args.stride, "cortes": n,
               "ms_sam_por_corte": round(1000 * sam_time / max(n, 1), 1), "regiones": {}}
    for name, methods in rows.items():
        summary["regiones"][name] = {
            method: {"dice": round(float(np.mean([v[0] for v in vals])), 4),
                     "iou": round(float(np.mean([v[1] for v in vals])), 4), "n": len(vals)}
            for method, vals in methods.items() if vals}
    for method in ("sam", "nuestro"):
        all_vals = [v for m in rows.values() for v in m[method]]
        summary[f"global_{method}"] = {"dice": round(float(np.mean([v[0] for v in all_vals])), 4),
                                       "iou": round(float(np.mean([v[1] for v in all_vals])), 4),
                                       "n": len(all_vals)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
