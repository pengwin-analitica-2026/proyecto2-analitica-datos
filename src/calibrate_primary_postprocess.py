"""Calibra el postproceso primario/secundario sobre validación (sin reentrenar).

Barre min_size, fill_holes y absorb_small sobre las predicciones de la cabeza de
instancia y reporta Dice/IoU emparejados. Se calibra en validación y nunca en
test, igual que calibrate_instance_boundary.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instance_postprocess import primary_secondary_to_instances
from pengwin_metrics import matched_instance_metrics
from week10_multitask import INSTANCE_MODES, PengwinMultiTaskModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache/week10_multitask_slices_x12.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/week10_primary_postprocess_calibration.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("instance_mode", "ids") != "primary":
        raise SystemExit("El checkpoint no es de modo primario/secundario.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PengwinMultiTaskModel(base_channels=int(checkpoint["base_channels"]),
                                  instance_mode="primary")
    model.load_state_dict(checkpoint["model"]); model.to(device).eval()

    data = np.load(args.cache)
    mask = data["split_names"] == args.split
    images = torch.from_numpy(data["images"][mask].astype(np.float32))
    labels = data["labels"][mask].astype(np.int64)
    ps, semantic = [], []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            out = model(images[start:start + args.batch_size].to(device))
            ps.append(out["instance"].argmax(1).cpu().numpy())
            semantic.append(out["semantic"].argmax(1).cpu().numpy())
    ps, semantic = np.concatenate(ps), np.concatenate(semantic)

    rows = []
    grid = itertools.product([1, 4, 8, 12, 20, 32, 48, 64], [False, True], [0, 16, 32, 64, 96])
    for min_size, fill_holes, absorb_small in grid:
        prediction = primary_secondary_to_instances(ps, semantic, min_size=min_size,
                                                    fill_holes=fill_holes,
                                                    absorb_small=absorb_small)
        rows.append({"min_size": min_size, "fill_holes": fill_holes,
                     "absorb_small": absorb_small,
                     **matched_instance_metrics(prediction, labels)})
    rows.sort(key=lambda row: row["instance_dice_matched"], reverse=True)
    baseline = next(r for r in rows if r["min_size"] == 12 and not r["fill_holes"] and r["absorb_small"] == 0)
    print(f"Baseline (min_size=12, sin relleno, sin absorción): dice={baseline['instance_dice_matched']:.4f} "
          f"iou={baseline['instance_iou_matched']:.4f}")
    print("Top 10:")
    for row in rows[:10]:
        print(f"  min_size={row['min_size']:>3} fill_holes={str(row['fill_holes']):5} "
              f"absorb_small={row['absorb_small']:>3} → dice={row['instance_dice_matched']:.4f} "
              f"iou={row['instance_iou_matched']:.4f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"baseline": baseline, "best": rows[0], "rows": rows}, indent=2))
    print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
