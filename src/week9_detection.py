"""Semana 9: backbone propio, CBAM y deteccion 2D por grid.

La cabeza predice por celda:
    objectness, (cx, cy, w, h) normalizados y 3 clases anatómicas.

No usa YOLO, torchvision ni otro framework de detección. La salida es
compatible con el pipeline corte-a-corte que se construirá en semanas 10-11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


NUM_CLASSES = 3
REGION_NAMES = ("sacro", "coxal_izquierdo", "coxal_derecho")


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Atención de canal seguida de atención espacial."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial(self.channel(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FundidoraBackbone(nn.Module):
    """Backbone convolucional pequeño y entrenable desde cero.

    Produce un mapa 16 veces menor que la imagen. CBAM se aplica al final,
    antes de la bifurcación hacia las cabezas de tareas.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(in_channels, c, 2),
            ConvBlock(c, c * 2, 2),
            ConvBlock(c * 2, c * 4, 2),
            ConvBlock(c * 4, c * 8, 2),
        )
        self.cbam = CBAM(c * 8)
        self.out_channels = c * 8

    def forward(self, x: Tensor) -> Tensor:
        return self.cbam(self.features(x))


class GridDetectionHead(nn.Module):
    """Cabeza propia: una predicción por celda del grid."""

    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 5 + num_classes, 1),
        )
        # Inicialización conservadora: al principio casi no hay objetos.
        nn.init.constant_(self.predictor[-1].bias, 0.0)

    def forward(self, features: Tensor) -> Tensor:
        # B,C,H,W -> B,H,W,5+classes
        return self.predictor(features).permute(0, 2, 3, 1).contiguous()


class DetectionModel(nn.Module):
    """Modelo Semana 9: backbone + CBAM + cabeza de detección."""

    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32):
        super().__init__()
        self.backbone = FundidoraBackbone(base_channels=base_channels)
        self.detection_head = GridDetectionHead(self.backbone.out_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return self.detection_head(self.backbone(x))


@dataclass
class DetectionLossWeights:
    objectness: float = 1.0
    box: float = 5.0
    classification: float = 1.0


def detection_loss(pred: Tensor, target: Tensor,
                   weights: DetectionLossWeights | None = None) -> tuple[Tensor, dict[str, float]]:
    """Pérdida multitarea estable para el overfit de Semana 9.

    target y pred tienen forma [B, H, W, 5+C]. target contiene:
    [obj, cx, cy, w, h, one-hot-clases].
    """
    weights = weights or DetectionLossWeights()
    obj_target = target[..., 0]
    obj_logit = pred[..., 0]
    pos = obj_target > 0.5
    neg = ~pos
    # Reduce el dominio negativo para que el fondo no domine el overfit.
    obj_raw = F.binary_cross_entropy_with_logits(obj_logit, obj_target, reduction="none")
    obj_weight = torch.where(pos, torch.ones_like(obj_raw), torch.full_like(obj_raw, 0.25))
    obj_loss = (obj_raw * obj_weight).mean()

    if pos.any():
        box_loss = F.smooth_l1_loss(torch.sigmoid(pred[..., 1:5])[pos], target[..., 1:5][pos])
        cls_loss = F.cross_entropy(pred[..., 5:][pos], target[..., 5:][pos].argmax(dim=-1))
    else:
        box_loss = pred[..., 1:5].sum() * 0.0
        cls_loss = pred[..., 5:].sum() * 0.0

    total = (weights.objectness * obj_loss + weights.box * box_loss +
             weights.classification * cls_loss)
    values = {"total": float(total.detach()), "objectness": float(obj_loss.detach()),
              "box": float(box_loss.detach()), "classification": float(cls_loss.detach()),
              "positive_cells": float(pos.sum().detach())}
    return total, values


def labels_to_grid(label: Tensor, num_classes: int = NUM_CLASSES) -> Tensor:
    """Convierte máscaras [B,H,W] en targets [B,h,w,5+C].

    Las cajas se calculan por región anatómica. Si una región aparece en el
    corte, se usa una sola caja que cubre todos sus fragmentos visibles.
    """
    if label.ndim != 3:
        raise ValueError(f"label debe ser [B,H,W], recibido {tuple(label.shape)}")
    b, height, width = label.shape
    gh, gw = height // 16, width // 16
    target = torch.zeros((b, gh, gw, 5 + num_classes), dtype=torch.float32, device=label.device)
    for bi in range(b):
        for cls in range(num_classes):
            if cls == 0:
                mask = (label[bi] >= 1) & (label[bi] <= 10)
            elif cls == 1:
                mask = (label[bi] >= 11) & (label[bi] <= 20)
            else:
                mask = (label[bi] >= 21) & (label[bi] <= 30)
            ys, xs = torch.where(mask)
            if len(xs) == 0:
                continue
            xmin, xmax = xs.min().float(), xs.max().float()
            ymin, ymax = ys.min().float(), ys.max().float()
            cx = ((xmin + xmax) / 2 + 0.5) / width
            cy = ((ymin + ymax) / 2 + 0.5) / height
            bw = (xmax - xmin + 1) / width
            bh = (ymax - ymin + 1) / height
            gi = min(int(cy * gh), gh - 1)
            gj = min(int(cx * gw), gw - 1)
            # Si hay colisión, conserva la caja de mayor área y la reporta.
            area = bw * bh
            old_area = target[bi, gi, gj, 3] * target[bi, gi, gj, 4]
            if target[bi, gi, gj, 0] == 1 and old_area >= area:
                continue
            target[bi, gi, gj, 0] = 1
            # cx/cy se guardan relativos a la celda, igual que los logits
            # decodificados con sigmoid. w/h permanecen normalizados a la imagen.
            tx = cx * gw - gj
            ty = cy * gh - gi
            target[bi, gi, gj, 1:5] = torch.stack([tx, ty, bw, bh])
            target[bi, gi, gj, 5:] = 0
            target[bi, gi, gj, 5 + cls] = 1
    return target


def box_iou_xyxy(a: Tensor, b: Tensor) -> Tensor:
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp(min=0).prod(dim=-1)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
    """NMS propia, sin torchvision."""
    keep: list[int] = []
    order = scores.argsort(descending=True)
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        ious = box_iou_xyxy(boxes[current:current + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def decode_predictions(pred: Tensor, confidence: float = 0.25,
                       iou_threshold: float = 0.5) -> list[list[dict[str, object]]]:
    """Decodifica la salida y aplica NMS independiente por clase."""
    if pred.ndim != 4:
        raise ValueError("pred debe tener forma [B,grid_y,grid_x,5+C]")
    b, gh, gw, channels = pred.shape
    num_classes = channels - 5
    result: list[list[dict[str, object]]] = []
    for bi in range(b):
        rows: list[dict[str, object]] = []
        for i in range(gh):
            for j in range(gw):
                objectness = torch.sigmoid(pred[bi, i, j, 0])
                class_prob, cls = torch.softmax(pred[bi, i, j, 5:], dim=-1).max(dim=-1)
                score = objectness * class_prob
                if float(score) < confidence:
                    continue
                cx = (torch.sigmoid(pred[bi, i, j, 1]) + j) / gw
                cy = (torch.sigmoid(pred[bi, i, j, 2]) + i) / gh
                w, h = torch.sigmoid(pred[bi, i, j, 3:5])
                box = torch.stack([(cx - w / 2).clamp(0, 1), (cy - h / 2).clamp(0, 1),
                                   (cx + w / 2).clamp(0, 1), (cy + h / 2).clamp(0, 1)])
                rows.append({"box": box, "score": score, "class_id": int(cls)})
        final: list[dict[str, object]] = []
        for cls in range(num_classes):
            candidates = [r for r in rows if r["class_id"] == cls]
            if not candidates:
                continue
            boxes = torch.stack([r["box"] for r in candidates])
            scores = torch.stack([r["score"] for r in candidates])
            for idx in nms(boxes, scores, iou_threshold).tolist():
                item = candidates[idx].copy()
                item["box"] = boxes[idx].detach().cpu().tolist()
                item["score"] = float(scores[idx].detach())
                final.append(item)
        result.append(sorted(final, key=lambda r: float(r["score"]), reverse=True))
    return result


def boxes_from_label(label: Tensor) -> list[list[dict[str, object]]]:
    """Extrae las cajas ground truth por región desde máscaras [B,H,W]."""
    if label.ndim != 3:
        raise ValueError("label debe tener forma [B,H,W]")
    batch: list[list[dict[str, object]]] = []
    for image_label in label:
        height, width = image_label.shape
        rows: list[dict[str, object]] = []
        for cls in range(NUM_CLASSES):
            low = 1 + 10 * cls
            high = low + 9
            ys, xs = torch.where((image_label >= low) & (image_label <= high))
            if len(xs) == 0:
                continue
            rows.append({
                "class_id": cls,
                "box": [float(xs.min() / width), float(ys.min() / height),
                        float((xs.max() + 1) / width), float((ys.max() + 1) / height)],
            })
        batch.append(rows)
    return batch
