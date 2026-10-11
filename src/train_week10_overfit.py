"""Overfit conjunto de clasificación, detección y segmentación de Semana 10."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week9_detection import labels_to_grid
from week10_multitask import (PengwinMultiTaskModel, labels_to_semantic,
                               multitask_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache/week9_detection_slices.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="outputs/week10_multitask_overfit.pth")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    data = np.load(args.cache)
    x = torch.from_numpy(data["images"][:args.samples].astype(np.float32))
    labels = torch.from_numpy(data["labels"][:args.samples].astype(np.int64))
    target_detection = labels_to_grid(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, labels, target_detection = x.to(device), labels.to(device), target_detection.to(device)
    model = PengwinMultiTaskModel(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    first, last = None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(x)
            loss, parts = multitask_loss(outputs, target_detection, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        first = parts["total"] if first is None else first
        last = parts["total"]
        if epoch == 1 or epoch % max(args.epochs // 10, 1) == 0:
            print(f"epoch {epoch:04d} total={parts['total']:.5f} "
                  f"det={parts['detection']:.5f} cls={parts['classification']:.5f} "
                  f"sem={parts['semantic']:.5f} inst={parts['instance']:.5f}")
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        semantic_target = labels_to_semantic(labels)
        instance_prediction = outputs["instance"].argmax(1)
        semantic_accuracy = (outputs["semantic"].argmax(1) == semantic_target).float().mean()
        instance_accuracy = (instance_prediction == labels).float().mean()
        foreground = labels > 0
        instance_foreground_accuracy = (
            (instance_prediction[foreground] == labels[foreground]).float().mean()
            if foreground.any() else torch.tensor(0.0, device=labels.device)
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                "input_size": int(data["size"]), "samples": args.samples,
                "semantic_accuracy": float(semantic_accuracy),
                "instance_accuracy": float(instance_accuracy),
                "instance_foreground_accuracy": float(instance_foreground_accuracy)}, args.output)
    print(f"\nDispositivo: {device} | AMP: {amp_enabled}")
    print(f"Pérdida total: {first:.5f} -> {last:.5f} | reducción {first / max(last, 1e-8):.1f}x")
    print(f"Accuracy semántica: {float(semantic_accuracy):.4f} | "
          f"accuracy instancia: {float(instance_accuracy):.4f} | "
          f"foreground: {float(instance_foreground_accuracy):.4f}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
