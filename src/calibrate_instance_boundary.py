"""Calibra postproceso de instancias sin volver a entrenar ni tocar test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instance_postprocess import semantic_to_instances
from pengwin_metrics import matched_instance_metrics
from week10_multitask import PengwinMultiTaskModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache/week10_multitask_slices.npz")
    parser.add_argument("--checkpoint", default="outputs/week10_multitask_instance_boundary.pth")
    parser.add_argument("--output", default="outputs/week10_instance_boundary_calibration.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    data = np.load(args.cache)
    mask = data["split_names"] == "val"
    images = torch.from_numpy(data["images"][mask].astype(np.float32))
    labels = data["labels"][mask]
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = PengwinMultiTaskModel(base_channels=int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    semantics, boundaries = [], []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            output = model(images[start:start + args.batch_size])
            semantics.append(output["semantic"].argmax(1).numpy())
            boundaries.append(torch.sigmoid(output["boundary"]).squeeze(1).numpy())
    semantic = np.concatenate(semantics)
    boundary = np.concatenate(boundaries)
    rows = []
    baseline = matched_instance_metrics(semantic_to_instances(semantic), labels)
    rows.append({"threshold": None, "min_size": 12, **baseline})
    for threshold in (0.70, 0.80, 0.90, 0.95, 0.98, 0.99):
        for min_size in (4, 8, 12, 24):
            prediction = semantic_to_instances(semantic, boundary, threshold, min_size)
            rows.append({"threshold": threshold, "min_size": min_size,
                         **matched_instance_metrics(prediction, labels)})
    rows.sort(key=lambda row: row["instance_dice_matched"], reverse=True)
    result = {"best": rows[0], "trials": rows}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["best"], indent=2))


if __name__ == "__main__":
    main()
