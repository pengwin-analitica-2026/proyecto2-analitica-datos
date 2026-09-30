"""Primer entrenamiento de detección multi-paciente para Semana 9."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week9_detection import (DetectionModel, box_iou_xyxy, boxes_from_label,
                              decode_predictions, detection_loss, labels_to_grid)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detection_iou(model: DetectionModel, images: torch.Tensor, labels: torch.Tensor,
                  device: torch.device, confidence: float = 0.20) -> tuple[float, int]:
    """IoU media por caja GT, tras NMS y comparando solo la misma clase."""
    model.eval()
    total, matched = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(images), 16):
            x = images[start:start + 16].to(device)
            predictions = decode_predictions(model(x), confidence=confidence)
            truths = boxes_from_label(labels[start:start + 16])
            for predicted, truth in zip(predictions, truths):
                for gt in truth:
                    same_class = [p for p in predicted if p["class_id"] == gt["class_id"]]
                    if not same_class:
                        matched += 1
                        continue
                    gt_box = torch.tensor([gt["box"]])
                    pred_boxes = torch.tensor([p["box"] for p in same_class])
                    total += float(box_iou_xyxy(gt_box, pred_boxes).max())
                    matched += 1
    return total / max(matched, 1), matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache/week9_detection_slices.npz")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="outputs/week9_detection_initial.pth")
    args = parser.parse_args()
    seed_everything()
    torch.set_num_threads(args.threads)
    data = np.load(args.cache)
    images = torch.from_numpy(data["images"].astype(np.float32))
    labels = torch.from_numpy(data["labels"].astype(np.int64))
    names = data["split_names"]
    train_mask = np.asarray(names == "train")
    val_mask = np.asarray(names == "val")
    train_x, train_y = images[train_mask], labels[train_mask]
    val_x, val_y = images[val_mask], labels[val_mask]
    train_target = labels_to_grid(train_y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(TensorDataset(train_x, train_target), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    model = DetectionModel(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_iou, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = detection_loss(model(x.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_iou, n_boxes = detection_iou(model, val_x, val_y, device)
        if val_iou > best_iou:
            best_iou = val_iou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} train_loss={np.mean(losses):.4f} val_iou={val_iou:.4f} "
                  f"val_boxes={n_boxes}")
    model.load_state_dict(best_state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                "input_size": int(data["size"]), "cache": str(args.cache),
                "best_val_iou": best_iou}, args.output)
    print(f"Mejor IoU de validación: {best_iou:.4f}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()

