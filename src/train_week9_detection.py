"""Primer entrenamiento de deteccion multi-paciente para Semana 9.

Uso (desde la raiz del repo, con el cache ya generado):
    python src/train_week9_detection.py --output outputs/week9_det_cbam32.pth
    python src/train_week9_detection.py --no-cbam --output outputs/week9_det_nocbam32.pth

SELECCION DEL CHECKPOINT: por F1 de deteccion (IoU >= 0.5), no por IoU medio.
El IoU medio solo mira la mejor prediccion de cada caja real e ignora las que
sobran; en la primera corrida con gamma eligio una epoca que predecia 2112
cajas para 164 reales (precision 3 %). F1 combina precision y exhaustividad,
asi que un modelo que dispara cajas por todos lados no puede ganar.

ANCHO POR DEFECTO = 32 canales base. Con 8 canales el modelo sobreajusta los
96 cortes de entrenamiento y su mejor F1 fue 0.31 (con CBAM); con 32 llego a
0.56 en las mismas condiciones. El ancho era el cuello de botella principal.

Ademas del checkpoint se guarda el historial por epoca en un JSON junto a el
(mismo nombre, extension .json), para graficar las curvas en el informe.
"""

from __future__ import annotations

import argparse
import json
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
                      iou_threshold: float = 0.5,
                      max_por_clase: int | None = None) -> dict[str, float]:
    """Metricas de deteccion sobre validacion, tras NMS y por clase.

    iou_medio  mejor IoU por caja real (0 si no se detecto). Se conserva para
               comparar con el resultado documentado antes de la asesoria,
               pero NO castiga las cajas que sobran.
    precision  de las cajas predichas, cuantas aciertan una real (IoU >= 0.5).
    recall     de las cajas reales, cuantas fueron encontradas.
    f1         media armonica de las dos: criterio de seleccion.

    max_por_clase se pasa a decode_predictions (None = todas las cajas tras
    NMS, que es como se seleccionaron los checkpoints de la semana 9).
    """
    model.eval()
    acumulado = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n_gt": 0.0,
                 "n_pred": 0.0, "iou_suma": 0.0}
    with torch.no_grad():
        for start in range(0, len(images), 16):
            x = images[start:start + 16].to(device)
            predictions = decode_predictions(model(x), confidence=confidence,
                                             max_por_clase=max_por_clase)
            truths = boxes_from_label(labels[start:start + 16])
            for predicted, truth in zip(predictions, truths):
                for clave, valor in match_detections(predicted, truth, iou_threshold).items():
                    acumulado[clave] += valor

    tp, fp, fn = acumulado["tp"], acumulado["fp"], acumulado["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou_medio": acumulado["iou_suma"] / max(acumulado["n_gt"], 1),
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
    parser.add_argument("--base-channels", type=int, default=32)
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
    print(f"Modelo: {args.base_channels} canales base, CBAM "
          f"{'si' if args.use_cbam else 'no (rama de ablacion)'}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(TensorDataset(train_x, train_target), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    model = DetectionModel(base_channels=args.base_channels,
                           use_cbam=args.use_cbam).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    historial: list[dict] = []
    best_f1, best_iou, best_epoch, best_state, best_metrics = -1.0, -1.0, 0, None, {}
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
        historial.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})

        # F1 manda; el IoU medio solo desempata.
        mejora = (metrics["f1"] > best_f1 or
                  (metrics["f1"] == best_f1 and metrics["iou_medio"] > best_iou))
        if mejora:
            best_f1, best_iou, best_epoch = metrics["f1"], metrics["iou_medio"], epoch
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs or mejora:
            print(f"epoch {epoch:03d} train_loss={np.mean(losses):.4f} "
                  f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f} FP={int(metrics['falsos_positivos'])} "
                  f"iou={metrics['iou_medio']:.3f}" + ("  <- mejor" if mejora else ""))

    model.load_state_dict(best_state)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                "input_size": int(data["size"]), "cache": str(args.cache),
                "use_cbam": args.use_cbam, "best_epoch": best_epoch,
                "best_val_f1": best_f1, "best_val_iou": best_iou,
                "val_metrics": best_metrics, "target_encoding": codificacion}, out)
    historial_path = out.with_suffix(".json")
    historial_path.write_text(json.dumps({
        "config": {"base_channels": args.base_channels, "use_cbam": args.use_cbam,
                   "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                   "seed": 42, "criterio_seleccion": "f1@iou0.5"},
        "mejor_epoca": best_epoch,
        "historial": historial,
    }, indent=2))

    print(f"\nMejor epoca: {best_epoch} (seleccion por F1)")
    print(f"  F1={best_metrics['f1']:.4f}  precision={best_metrics['precision']:.4f}  "
          f"recall={best_metrics['recall']:.4f}")
    print(f"  falsos positivos={int(best_metrics['falsos_positivos'])}  "
          f"cajas reales={int(best_metrics['n_gt'])}  predichas={int(best_metrics['n_pred'])}")
    print(f"  IoU medio={best_metrics['iou_medio']:.4f}")
    print(f"Checkpoint: {out}")
    print(f"Historial : {historial_path}")


if __name__ == "__main__":
    main()