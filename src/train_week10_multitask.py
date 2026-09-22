"""Entrenamiento/evaluación completo de las tres cabezas PENGWIN."""

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
from instance_postprocess import semantic_to_instances
from pengwin_metrics import (detection_metrics, matched_instance_metrics,
                             multilabel_f1_auc, refine_boxes_with_semantic,
                             segmentation_metrics)
from week9_detection import boxes_from_label, decode_predictions, labels_to_grid
from week10_multitask import (MultiTaskLossWeights, PengwinMultiTaskModel,
                              labels_to_presence, labels_to_semantic,
                              multitask_loss)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def evaluate(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor,
             device: torch.device, batch_size: int, confidence: float = 0.10,
             use_boundary: bool = False, boundary_threshold: float = 0.50) -> dict[str, float]:
    model.eval()
    all_logits, all_targets, all_predictions, all_truths = [], [], [], []
    all_instances, all_direct_instances, all_semantic, all_labels = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size].to(device)
            y = labels[start:start + batch_size]
            output = model(x)
            all_logits.append(output["classification"].cpu().numpy())
            all_targets.append(labels_to_presence(y).numpy())
            all_predictions.extend(decode_predictions(output["detection"], confidence=confidence))
            all_truths.extend(boxes_from_label(y))
            boundary = (torch.sigmoid(output["boundary"]).squeeze(1).cpu().numpy()
                        if use_boundary else None)
            all_instances.append(semantic_to_instances(
                output["semantic"].argmax(1).cpu().numpy(), boundary,
                boundary_threshold=boundary_threshold))
            all_direct_instances.append(output["instance"].argmax(1).cpu().numpy())
            all_semantic.append(output["semantic"].argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    semantic_predictions = np.concatenate(all_semantic)
    result = multilabel_f1_auc(np.concatenate(all_logits), np.concatenate(all_targets))
    result.update(detection_metrics(refine_boxes_with_semantic(all_predictions, semantic_predictions), all_truths))
    result.update(segmentation_metrics(semantic_predictions,
                                       labels_to_semantic(torch.from_numpy(np.concatenate(all_labels))).numpy()))
    result.update(matched_instance_metrics(np.concatenate(all_instances), np.concatenate(all_labels)))
    direct = matched_instance_metrics(np.concatenate(all_direct_instances), np.concatenate(all_labels))
    result["instance_direct_dice_matched"] = direct["instance_dice_matched"]
    result["instance_direct_iou_matched"] = direct["instance_iou_matched"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache/week10_multitask_slices.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evalúa validación cada N épocas; reduce coste CPU.")
    parser.add_argument("--output", default="outputs/week10_multitask_best.pth")
    parser.add_argument("--metrics-output", default="outputs/week10_validation_metrics.json")
    parser.add_argument("--resume", default=None, help="Checkpoint para fine-tuning.")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--use-boundary", action="store_true")
    parser.add_argument("--boundary-threshold", type=float, default=0.50)
    parser.add_argument("--detection-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=2.0)
    parser.add_argument("--semantic-dice-weight", type=float, default=1.0)
    parser.add_argument("--instance-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--freeze-except-boundary", action="store_true")
    args = parser.parse_args()
    seed_everything(); torch.set_num_threads(args.threads)
    data = np.load(args.cache)
    images = torch.from_numpy(data["images"].astype(np.float32))
    labels = torch.from_numpy(data["labels"].astype(np.int64))
    names = data["split_names"]
    train_mask, val_mask = names == "train", names == "val"
    train_x, train_y = images[train_mask], labels[train_mask]
    val_x, val_y = images[val_mask], labels[val_mask]
    train_targets = labels_to_grid(train_y)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    loader = DataLoader(TensorDataset(train_x, train_y, train_targets), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    model = PengwinMultiTaskModel(base_channels=args.base_channels).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Pesos inesperados al reanudar: {unexpected}")
        if missing:
            print(f"Capas nuevas inicializadas: {missing}")
    if args.freeze_except_boundary:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.segmentation_head.boundary.parameters():
            parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_score, best_state, best_metrics = -float("inf"), None, None
    loss_weights = MultiTaskLossWeights(
        detection=args.detection_weight,
        classification=args.classification_weight,
        semantic=args.semantic_weight,
        instance=args.instance_weight,
        semantic_dice=args.semantic_dice_weight,
        boundary=args.boundary_weight,
    )
    if args.epochs == 0:
        best_metrics = evaluate(model, val_x, val_y, device, args.batch_size, args.confidence,
                                args.use_boundary, args.boundary_threshold)
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_except_boundary:
            # Conserva estadísticas BatchNorm del checkpoint validado.
            model.eval()
            model.segmentation_head.boundary.train()
        losses = []
        for x, y, target in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss, _ = multitask_loss(model(x.to(device)), target.to(device), y.to(device),
                                         weights=loss_weights)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        should_evaluate = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            metrics = evaluate(model, val_x, val_y, device, args.batch_size, args.confidence,
                               args.use_boundary, args.boundary_threshold)
            score = metrics["segmentation_dice_macro"] + metrics["detection_map50"]
            if score > best_score:
                best_score, best_metrics = score, metrics
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if should_evaluate:
            print(f"epoch {epoch:03d} loss={np.mean(losses):.4f} "
                  f"dice={metrics['segmentation_dice_macro']:.4f} "
                  f"map50={metrics['detection_map50']:.4f} "
                  f"iou_bbox={metrics['detection_iou_mean']:.4f}")
    model.load_state_dict(best_state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                "input_size": int(data["size"]), "best_validation": best_metrics}, args.output)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).write_text(json.dumps(best_metrics, indent=2))
    print("Mejores métricas de validación:", json.dumps(best_metrics, indent=2))
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
