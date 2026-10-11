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
from instance_postprocess import (geometry_to_instances, primary_secondary_to_instances,
                                  semantic_to_instances)
from pengwin_metrics import (detection_metrics, matched_instance_metrics,
                             multilabel_f1_auc, refine_boxes_with_semantic,
                             segmentation_metrics)
from week9_detection import boxes_from_label, decode_predictions, labels_to_grid
from week10_multitask import (INSTANCE_MODES, MultiTaskLossWeights, PengwinMultiTaskModel,
                              flip_sagittal, fracture_distance_weight, labels_to_presence, labels_to_semantic,
                              multitask_loss, region_crops, two_pass_instances)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def evaluate(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor,
             device: torch.device, batch_size: int, confidence: float = 0.10,
             use_boundary: bool = False, boundary_threshold: float = 0.50,
             use_geometry: bool = False, center_threshold: float = 0.25,
             two_pass: bool = False) -> dict[str, float]:
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
            semantic_batch = output["semantic"].argmax(1).cpu().numpy()
            if use_geometry:
                all_instances.append(geometry_to_instances(
                    semantic_batch, torch.sigmoid(output["centers"]).cpu().numpy(),
                    output["offsets"].cpu().numpy(), center_threshold=center_threshold))
            else:
                all_instances.append(semantic_to_instances(
                    semantic_batch, boundary, boundary_threshold=boundary_threshold))
            if two_pass and output["instance"].shape[1] == INSTANCE_MODES["primary"]:
                instance_argmax = two_pass_instances(model, x, output["semantic"].argmax(1)).cpu().numpy()
            else:
                instance_argmax = output["instance"].argmax(1).cpu().numpy()
            if output["instance"].shape[1] == INSTANCE_MODES["primary"]:
                # Modo primario/secundario: la cabeza de instancia se convierte a IDs
                # con componentes conectados y se restringe a la semántica.
                instance_argmax = primary_secondary_to_instances(instance_argmax, semantic_batch)
            all_direct_instances.append(instance_argmax)
            all_semantic.append(output["semantic"].argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    semantic_predictions = np.concatenate(all_semantic)
    result = multilabel_f1_auc(np.concatenate(all_logits), np.concatenate(all_targets))
    result.update(detection_metrics(refine_boxes_with_semantic(all_predictions, semantic_predictions), all_truths))
    result.update(segmentation_metrics(semantic_predictions,
                                       labels_to_semantic(torch.from_numpy(np.concatenate(all_labels))).numpy()))
    semantic_cc = matched_instance_metrics(np.concatenate(all_instances), np.concatenate(all_labels))
    direct = matched_instance_metrics(np.concatenate(all_direct_instances), np.concatenate(all_labels))
    if model.instance_mode == "primary":
        # La métrica principal de instancia sale de la cabeza primario/secundario;
        # la conectividad sobre la semántica se conserva como referencia.
        result.update(direct)
        result["instance_semantic_cc_dice_matched"] = semantic_cc["instance_dice_matched"]
        result["instance_semantic_cc_iou_matched"] = semantic_cc["instance_iou_matched"]
    else:
        result.update(semantic_cc)
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
    parser.add_argument("--use-geometry", action="store_true")
    parser.add_argument("--center-threshold", type=float, default=0.25)
    parser.add_argument("--detection-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=2.0)
    parser.add_argument("--semantic-dice-weight", type=float, default=1.0)
    parser.add_argument("--instance-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--freeze-except-boundary", action="store_true")
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--instance-dice-weight", type=float, default=1.0)
    parser.add_argument("--freeze-except-geometry", action="store_true")
    parser.add_argument("--instance-mode", choices=sorted(INSTANCE_MODES), default="ids",
                        help="ids: 31 IDs PENGWIN (original). primary: primario/secundario "
                             "por región (formulación del challenge PENGWIN 2024).")
    parser.add_argument("--augment", action="store_true",
                        help="Flip sagital aleatorio (50%%) con intercambio de coxales "
                             "y jitter leve de intensidad.")
    parser.add_argument("--crop-prob", type=float, default=0.0,
                        help="Fracción de cada batch a la que se añaden recortes por región "
                             "(FracSegNet Step 2). 0 desactiva.")
    parser.add_argument("--two-pass", action="store_true",
                        help="Evalúa la instancia con segundo pase por región recortada.")
    parser.add_argument("--cosine", action="store_true",
                        help="Cosine annealing de la tasa de aprendizaje hasta lr/50 en la "
                             "última época; reduce la oscilación de validación al final.")
    parser.add_argument("--secondary-weight", type=float, default=1.0,
                        help="Peso CE de las clases secundario en modo primary (1 = sin cambio).")
    parser.add_argument("--fracture-weight", type=float, default=0.0,
                        help="Peso extra (FracSegNet) junto a las líneas de fractura; 0 desactiva.")
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
    # El mapa de peso por distancia a la fractura depende solo de las etiquetas:
    # se calcula UNA vez por corte (antes se recalculaba en cada batch y época).
    if args.fracture_weight > 0:
        print("Precomputando mapas de distancia a la fractura...", flush=True)
        train_w = torch.cat([fracture_distance_weight(train_y[i:i + 64], args.fracture_weight)
                             for i in range(0, len(train_y), 64)])
    else:
        train_w = torch.ones(train_y.shape, dtype=torch.float32)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    loader = DataLoader(TensorDataset(train_x, train_y, train_targets, train_w), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(42),
                        pin_memory=torch.cuda.is_available())
    in_channels = int(images.shape[1])
    if in_channels > 1:
        print(f"Entrada 2.5D: {in_channels} canales (corte central ± {(in_channels - 1) // 2})")
    model = PengwinMultiTaskModel(base_channels=args.base_channels,
                                  instance_mode=args.instance_mode,
                                  in_channels=in_channels).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state = dict(checkpoint["model"])
        current = model.state_dict()
        # Permite partir del checkpoint de 31 IDs al modo primario/secundario:
        # backbone, CBAM, detección, clasificación y decoder se reutilizan; solo
        # la capa final de instancia (forma distinta) se inicializa de cero.
        reinitialized = [key for key, value in state.items()
                         if key in current and tuple(value.shape) != tuple(current[key].shape)]
        for key in reinitialized:
            state.pop(key)
        if reinitialized:
            print(f"Capas reinicializadas por cambio de forma: {reinitialized}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Pesos inesperados al reanudar: {unexpected}")
        if missing:
            print(f"Capas nuevas inicializadas: {missing}")
    if args.freeze_except_boundary:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.segmentation_head.boundary.parameters():
            parameter.requires_grad = True
    if args.freeze_except_geometry:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for head in (model.segmentation_head.centers, model.segmentation_head.offsets):
            for parameter in head.parameters():
                parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1),
                                                            eta_min=args.lr / 50)
                 if args.cosine else None)
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
        centers=args.center_weight,
        offsets=args.offset_weight,
        instance_dice=args.instance_dice_weight,
        fracture_weight=args.fracture_weight,
        secondary_weight=args.secondary_weight,
    )
    if args.epochs == 0:
        best_metrics = evaluate(model, val_x, val_y, device, args.batch_size, args.confidence,
                                args.use_boundary, args.boundary_threshold,
                                args.use_geometry, args.center_threshold, args.two_pass)
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_except_boundary or args.freeze_except_geometry:
            # Conserva estadísticas BatchNorm del checkpoint validado.
            model.eval()
            model.segmentation_head.boundary.train()
            if args.freeze_except_geometry:
                model.segmentation_head.centers.train()
                model.segmentation_head.offsets.train()
        losses = []
        for x, y, target, w in loader:
            if args.augment:
                x, y, w = x.clone(), y.clone(), w.clone()
                flip = torch.rand(x.shape[0]) < 0.5
                if flip.any():
                    x[flip], y[flip] = flip_sagittal(x[flip], y[flip])
                    w[flip] = torch.flip(w[flip], dims=(-1,))  # el mapa se espeja igual
                # Jitter de intensidad leve: robustez a la variabilidad de kernel de
                # reconstrucción observada en el EDA (caso 025).
                gain = 1.0 + 0.1 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5)
                bias = 0.05 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5)
                x = (x * gain + bias).clamp(0.0, 1.0)
                target = labels_to_grid(y)  # las cajas cambian con el flip
            if args.crop_prob > 0:
                pick = torch.rand(x.shape[0]) < args.crop_prob
                if pick.any():
                    cx, cy, boxes = region_crops(x[pick], y[pick])
                    if boxes:
                        cw = fracture_distance_weight(cy, args.fracture_weight)
                        x = torch.cat([x, cx]); y = torch.cat([y, cy]); w = torch.cat([w, cw])
                        target = labels_to_grid(y)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss, _ = multitask_loss(model(x.to(device, non_blocking=True)),
                                         target.to(device), y.to(device),
                                         weights=loss_weights, fracture=w.to(device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        should_evaluate = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            metrics = evaluate(model, val_x, val_y, device, args.batch_size, args.confidence,
                               args.use_boundary, args.boundary_threshold,
                               args.use_geometry, args.center_threshold, args.two_pass)
            score = (metrics["segmentation_dice_macro"] + metrics["detection_map50"] +
                     metrics["instance_dice_matched"])
            if score > best_score:
                best_score, best_metrics = score, metrics
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            # Checkpoint parcial para reanudar si la sesión se corta.
            last_path = Path(args.output).with_name(Path(args.output).stem + "_last.pth")
            last_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                        "instance_mode": args.instance_mode, "two_pass": args.two_pass,
                        "in_channels": in_channels,
                        "epoch": epoch, "metrics": metrics}, last_path)
        if should_evaluate:
            print(f"epoch {epoch:03d} loss={np.mean(losses):.4f} "
                  f"dice={metrics['segmentation_dice_macro']:.4f} "
                  f"map50={metrics['detection_map50']:.4f} "
                  f"iou_bbox={metrics['detection_iou_mean']:.4f}")
    model.load_state_dict(best_state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                "instance_mode": args.instance_mode, "augment": args.augment, "cosine": args.cosine,
                "fracture_weight": args.fracture_weight, "secondary_weight": args.secondary_weight,
                "crop_prob": args.crop_prob, "two_pass": args.two_pass, "in_channels": in_channels,
                "input_size": int(data["size"]), "best_validation": best_metrics}, args.output)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).write_text(json.dumps(best_metrics, indent=2))
    print("Mejores métricas de validación:", json.dumps(best_metrics, indent=2))
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
