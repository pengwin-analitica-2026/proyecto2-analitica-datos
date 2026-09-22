"""Métricas reproducibles del proyecto, sin frameworks de detección externos."""

from __future__ import annotations

import numpy as np
import torch

from week9_detection import box_iou_xyxy


def refine_boxes_with_semantic(predictions: list[list[dict]], semantic: np.ndarray) -> list[list[dict]]:
    """Ajusta los límites de cajas NMS con la máscara anatómica predicha.

    El detector decide presencia, clase y score; la segmentación afina los
    bordes de la región, una segunda etapa permitida por el pipeline del curso.
    """
    refined_batch: list[list[dict]] = []
    for rows, image in zip(predictions, semantic):
        height, width = image.shape
        best_per_class: dict[int, dict] = {}
        for row in rows:
            cls = int(row["class_id"])
            candidate = image == cls + 1
            ys, xs = np.where(candidate)
            updated = dict(row)
            if len(xs):
                updated["box"] = [float(xs.min() / width), float(ys.min() / height),
                                  float((xs.max() + 1) / width), float((ys.max() + 1) / height)]
            if cls not in best_per_class or float(updated["score"]) > float(best_per_class[cls]["score"]):
                best_per_class[cls] = updated
        refined_batch.append(sorted(best_per_class.values(), key=lambda row: float(row["score"]), reverse=True))
    return refined_batch


def multilabel_f1_auc(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """F1 macro y AUC macro para presencia de las tres regiones."""
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = probabilities >= 0.5
    f1s, aucs = [], []
    for category in range(targets.shape[1]):
        truth = targets[:, category].astype(bool)
        pred = predictions[:, category]
        tp = np.logical_and(pred, truth).sum()
        fp = np.logical_and(pred, ~truth).sum()
        fn = np.logical_and(~pred, truth).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        positives, negatives = truth.sum(), (~truth).sum()
        if positives and negatives:
            order = np.argsort(probabilities[:, category])
            ranks = np.empty(len(order), dtype=float)
            ranks[order] = np.arange(1, len(order) + 1)
            aucs.append((ranks[truth].sum() - positives * (positives + 1) / 2) /
                        (positives * negatives))
    return {"classification_f1_macro": float(np.mean(f1s)),
            "classification_auc_macro": float(np.mean(aucs)) if aucs else float("nan")}


def _ap_for_class(predictions: list[list[dict]], truths: list[list[dict]], cls: int,
                  threshold: float) -> float:
    rows, total_gt = [], 0
    for image_index, (predicted, truth) in enumerate(zip(predictions, truths)):
        gt = [item for item in truth if int(item["class_id"]) == cls]
        total_gt += len(gt)
        rows.extend((float(item["score"]), image_index, item) for item in predicted
                    if int(item["class_id"]) == cls)
    if not total_gt:
        return float("nan")
    rows.sort(reverse=True, key=lambda row: row[0])
    matched: dict[int, set[int]] = {}
    tp, fp = [], []
    for _, image_index, item in rows:
        gt = [g for g in truths[image_index] if int(g["class_id"]) == cls]
        if not gt:
            tp.append(0); fp.append(1); continue
        ious = box_iou_xyxy(torch.tensor([item["box"]], dtype=torch.float32),
                            torch.tensor([g["box"] for g in gt], dtype=torch.float32))[0].numpy()
        best = int(ious.argmax())
        used = matched.setdefault(image_index, set())
        if ious[best] >= threshold and best not in used:
            used.add(best); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    if not rows:
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.r_[0.0, recall, 1.0]
    precision = np.r_[1.0, precision, 0.0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.trapz(precision, recall))


def detection_metrics(predictions: list[list[dict]], truths: list[list[dict]]) -> dict[str, float]:
    """IoU por GT y AP/mAP a los umbrales requeridos."""
    ious = []
    for predicted, truth in zip(predictions, truths):
        for gt in truth:
            candidates = [p for p in predicted if p["class_id"] == gt["class_id"]]
            if candidates:
                value = box_iou_xyxy(torch.tensor([gt["box"]], dtype=torch.float32),
                                     torch.tensor([p["box"] for p in candidates], dtype=torch.float32)).max()
                ious.append(float(value))
            else:
                ious.append(0.0)
    ap50 = [_ap_for_class(predictions, truths, cls, 0.50) for cls in range(3)]
    thresholds = np.arange(0.50, 0.96, 0.05)
    aps = [_ap_for_class(predictions, truths, cls, threshold)
           for threshold in thresholds for cls in range(3)]
    return {"detection_iou_mean": float(np.mean(ious)) if ious else 0.0,
            "detection_map50": float(np.nanmean(ap50)),
            "detection_map50_95": float(np.nanmean(aps))}


def segmentation_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Dice e IoU macro por fragmento GT visible, excluyendo fondo."""
    values_dice, values_iou = [], []
    for label in np.unique(truth):
        if label == 0:
            continue
        gt = truth == label
        predicted = prediction == label
        intersection = np.logical_and(gt, predicted).sum()
        denom = gt.sum() + predicted.sum()
        union = np.logical_or(gt, predicted).sum()
        values_dice.append(2 * intersection / max(denom, 1))
        values_iou.append(intersection / max(union, 1))
    return {"segmentation_dice_macro": float(np.mean(values_dice)) if values_dice else 0.0,
            "segmentation_iou_macro": float(np.mean(values_iou)) if values_iou else 0.0}


def matched_instance_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Dice/IoU por fragmento tras matching uno-a-uno dentro de su región.

    Las instancias predichas no heredan IDs del ground truth: se emparejan por
    máximo IoU. Predicciones extra no mejoran la puntuación y fragmentos GT sin
    correspondencia reciben cero.
    """
    dices, ious = [], []
    for predicted_image, truth_image in zip(prediction, truth):
        for low, high in ((1, 10), (11, 20), (21, 30)):
            gt_ids = [int(value) for value in np.unique(truth_image)
                      if low <= value <= high]
            pred_ids = [int(value) for value in np.unique(predicted_image)
                        if low <= value <= high]
            used: set[int] = set()
            for gt_id in gt_ids:
                gt = truth_image == gt_id
                best_id, best_iou = None, 0.0
                for pred_id in pred_ids:
                    if pred_id in used:
                        continue
                    candidate = predicted_image == pred_id
                    iou = np.logical_and(gt, candidate).sum() / max(np.logical_or(gt, candidate).sum(), 1)
                    if iou > best_iou:
                        best_id, best_iou = pred_id, iou
                if best_id is None:
                    dices.append(0.0); ious.append(0.0); continue
                used.add(best_id)
                candidate = predicted_image == best_id
                dices.append(2 * np.logical_and(gt, candidate).sum() / max(gt.sum() + candidate.sum(), 1))
                ious.append(best_iou)
    return {"instance_dice_matched": float(np.mean(dices)) if dices else 0.0,
            "instance_iou_matched": float(np.mean(ious)) if ious else 0.0}
