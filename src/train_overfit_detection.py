"""Prueba de correctitud Semana 9: overfit deliberado de detección.

Uso real (requiere data/raw y dependencias instaladas):
    python src/train_overfit_detection.py --case-id 001 --epochs 300

La prueba selecciona cortes con etiquetas, redimensiona a 256x256, construye
las cajas por región y verifica que la pérdida baje sobre el mismo batch.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pengwin_io import load_case, resize_label_slice, resize_slice, window_hu
from week9_detection import DetectionModel, detection_loss, labels_to_grid, decode_predictions


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batch(case_id: str, size: int = 256, max_slices: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    case = load_case(case_id, load_label=True)
    if case.label is None:
        raise RuntimeError(f"El caso {case_id} no tiene máscara")
    indices = np.where((case.label != 0).any(axis=(1, 2)))[0]
    if len(indices) == 0:
        raise RuntimeError(f"El caso {case_id} no tiene cortes con etiquetas")
    indices = indices[np.linspace(0, len(indices) - 1, min(max_slices, len(indices))).astype(int)]
    images, labels = [], []
    for z in indices:
        images.append(resize_slice(window_hu(case.image[z]), size))
        labels.append(resize_label_slice(case.label[z], size))
    x = torch.from_numpy(np.stack(images)[:, None]).float()
    y = torch.from_numpy(np.stack(labels)).long()
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="001")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--max-slices", type=int, default=2,
                        help="cortes repetidos durante el overfit intencional")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=8,
                        help="ancho del backbone para la prueba de correctitud")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", default="outputs/week9_overfit_detection.pth")
    parser.add_argument("--min-reduction", type=float, default=4.0,
                        help="factor mínimo initial_loss/final_loss; 0 desactiva la aserción")
    args = parser.parse_args()
    seed_everything()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, labels = make_batch(args.case_id, size=args.size, max_slices=args.max_slices)
    target = labels_to_grid(labels).to(device)
    x = x.to(device)
    model = DetectionModel(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    first, last = None, None
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = detection_loss(model(x), target)
        loss.backward()
        optimizer.step()
        first = metrics["total"] if first is None else first
        last = metrics["total"]
        if epoch == 1 or epoch % max(args.epochs // 10, 1) == 0:
            print(f"epoch {epoch:04d} loss={metrics['total']:.5f} "
                  f"obj={metrics['objectness']:.5f} box={metrics['box']:.5f} "
                  f"cls={metrics['classification']:.5f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "case_id": args.case_id,
                "input_size": args.size, "base_channels": args.base_channels,
                "max_slices": args.max_slices,
                "regions": ("sacro", "coxal_izquierdo", "coxal_derecho")}, args.output)
    detections = decode_predictions(model(x), confidence=0.5)
    print(f"\nDispositivo: {device}")
    print(f"Pérdida inicial: {first:.5f} | final: {last:.5f}")
    print(f"Reducción: {first / max(last, 1e-8):.1f}x")
    print(f"Predicciones del primer corte: {detections[0]}")
    if args.min_reduction > 0 and first / max(last, 1e-8) < args.min_reduction:
        raise SystemExit("FALLÓ: la pérdida no bajó lo suficiente en el overfit")
    print(f"OK: checkpoint guardado en {args.output}")


if __name__ == "__main__":
    main()
