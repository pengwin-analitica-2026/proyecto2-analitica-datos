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
from week9_detection import (DetectionModel, boxes_from_label, decode_predictions,
                              detection_loss, labels_to_grid, match_detections)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluar_deteccion(model: DetectionModel, images: torch.Tensor, labels: torch.Tensor,
                      device: torch.device, confidence: float = 0.20,
                      iou_threshold: float = 0.5) -> dict[str, float]:
    """Métricas de detección sobre validación, tras NMS y por clase.

    Reporta IoU medio por caja real (igual que antes, para poder comparar con
    el resultado documentado) y además precisión, exhaustividad y falsos
    positivos, que el IoU medio por sí solo no captura: sin ellos, un modelo
    que dispare muchas cajas parece mejor de lo que es.
    """
    model.eval()
    acumulado = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n_gt": 0.0,
                 "n_pred": 0.0, "iou_suma": 0.0}
    with torch.no_grad():
        for start in range(0, len(images), 16):
            x = images[start:start + 16].to(device)
            predictions = decode_predictions(model(x), confidence=confidence)
            truths = boxes_from_label(labels[start:start + 16])
            for predicted, truth in zip(predictions, truths):
                for clave, valor in match_detections(predicted, truth, iou_threshold).items():
                    acumulado[clave] += valor

    tp, fp, fn = acumulado["tp"], acumulado["fp"], acumulado["fn"]
    return {
        "iou_medio": acumulado["iou_suma"] / max(acumulado["n_gt"], 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "falsos_positivos": fp,
        "n_gt": acumulado["n_gt"],
        "n_pred": acumulado["n_pred"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache/week9_detection_slices.npz")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--no-cbam", dest="use_cbam", action="store_false",
                        help="rama de control de la ablacion: reemplaza CBAM por Identity")
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
    codificacion: dict = {}
    train_target = labels_to_grid(train_y, stats=codificacion)
    print(f"Codificacion del target: {codificacion.get('reubicadas', 0)} regiones reubicadas "
          f"por colision de celda, {codificacion.get('descartadas', 0)} descartadas")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(TensorDataset(train_x, train_target), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    model = DetectionModel(base_channels=args.base_channels,
                           use_cbam=args.use_cbam).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_iou, best_state, best_metrics = -1.0, None, {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = detection_loss(model(x.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluar_deteccion(model, val_x, val_y, device)
        if metrics["iou_medio"] > best_iou:
            best_iou = metrics["iou_medio"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} train_loss={np.mean(losses):.4f} "
                  f"val_iou={metrics['iou_medio']:.4f} P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f} FP={int(metrics['falsos_positivos'])} "
                  f"gt={int(metrics['n_gt'])}")
    model.load_state_dict(best_state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                "input_size": int(data["size"]), "cache": str(args.cache),
                "use_cbam": args.use_cbam, "best_val_iou": best_iou,
                "val_metrics": best_metrics,
                "target_encoding": codificacion}, args.output)
    print(f"\nCBAM: {'si' if args.use_cbam else 'no (rama de ablacion)'}")
    print(f"Mejor IoU de validacion: {best_iou:.4f}")
    print(f"  precision={best_metrics['precision']:.4f}  recall={best_metrics['recall']:.4f}  "
          f"falsos positivos={int(best_metrics['falsos_positivos'])}")
    print(f"  cajas reales={int(best_metrics['n_gt'])}  predichas={int(best_metrics['n_pred'])}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()