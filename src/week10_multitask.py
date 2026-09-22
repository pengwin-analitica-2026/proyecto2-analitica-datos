"""Arquitectura multitarea PENGWIN para Semana 10.

Un backbone propio con CBAM alimenta tres cabezas entrenadas desde cero:
clasificación anatómica, detección por grid y segmentación semántica/instancia.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from week9_detection import (DetectionLossWeights, FundidoraBackbone,
                              GridDetectionHead, detection_loss)


class ClassificationHead(nn.Module):
    """Presencia multietiqueta de sacro, coxal izquierdo y derecho."""

    def __init__(self, in_channels: int, num_regions: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, max(in_channels // 2, 8)),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(max(in_channels // 2, 8), num_regions),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(self.pool(features))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SegmentationHead(nn.Module):
    """Decoder común con salida semántica y salida de IDs de fragmento."""

    def __init__(self, in_channels: int, semantic_classes: int = 4,
                 instance_classes: int = 31):
        super().__init__()
        c1 = max(in_channels // 2, 8)
        c2 = max(in_channels // 4, 8)
        c3 = max(in_channels // 8, 8)
        self.decoder = nn.Sequential(
            UpsampleBlock(in_channels, c1),
            UpsampleBlock(c1, c2),
            UpsampleBlock(c2, c3),
            UpsampleBlock(c3, c3),
        )
        # Proyecciones aditivas: conservan el decoder anterior y agregan detalle
        # espacial de /8, /4 y /2 sin inflar mucho el número de parámetros.
        self.skip_projections = nn.ModuleList([
            nn.Conv2d(in_channels // 2, c1, 1, bias=False),
            nn.Conv2d(in_channels // 4, c2, 1, bias=False),
            nn.Conv2d(in_channels // 8, c3, 1, bias=False),
        ])
        self.semantic = nn.Conv2d(c3, semantic_classes, 1)
        self.instance = nn.Conv2d(c3, instance_classes, 1)
        self.boundary = nn.Conv2d(c3, 1, 1)

    def forward(self, features: Tensor, skips: list[Tensor] | None = None) -> tuple[Tensor, Tensor, Tensor]:
        decoded = features
        for index, block in enumerate(self.decoder):
            decoded = block(decoded)
            if skips is not None and index < 3:
                decoded = decoded + self.skip_projections[index](skips[-1 - index])
        return self.semantic(decoded), self.instance(decoded), self.boundary(decoded)


class PengwinMultiTaskModel(nn.Module):
    """Backbone+CBAM compartido y tres cabezas requeridas por el proyecto."""

    def __init__(self, base_channels: int = 8):
        super().__init__()
        self.backbone = FundidoraBackbone(base_channels=base_channels)
        channels = self.backbone.out_channels
        self.classification_head = ClassificationHead(channels)
        self.detection_head = GridDetectionHead(channels)
        self.segmentation_head = SegmentationHead(channels)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features, skips = self.backbone.forward_with_skips(x)
        semantic, instance, boundary = self.segmentation_head(features, skips)
        return {
            "classification": self.classification_head(features),
            "detection": self.detection_head(features),
            "semantic": semantic,
            "instance": instance,
            "boundary": boundary,
        }


def labels_to_semantic(labels: Tensor) -> Tensor:
    """IDs PENGWIN 0..30 → fondo/sacro/coxal izquierdo/coxal derecho."""
    semantic = torch.zeros_like(labels, dtype=torch.long)
    semantic[(labels >= 1) & (labels <= 10)] = 1
    semantic[(labels >= 11) & (labels <= 20)] = 2
    semantic[(labels >= 21) & (labels <= 30)] = 3
    return semantic


def labels_to_presence(labels: Tensor) -> Tensor:
    """Target multietiqueta [B,3] para regiones visibles en cada corte."""
    return torch.stack([
        ((labels >= 1) & (labels <= 10)).any(dim=(-2, -1)),
        ((labels >= 11) & (labels <= 20)).any(dim=(-2, -1)),
        ((labels >= 21) & (labels <= 30)).any(dim=(-2, -1)),
    ], dim=1).float()


@dataclass
class MultiTaskLossWeights:
    detection: float = 1.0
    classification: float = 0.5
    semantic: float = 2.0
    instance: float = 1.0
    semantic_dice: float = 1.0
    boundary: float = 1.0


def multiclass_foreground_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Dice por macro-región; evita que el fondo domine la segmentación."""
    probabilities = torch.softmax(logits, dim=1)
    losses = []
    for category in (1, 2, 3):
        predicted = probabilities[:, category]
        truth = (target == category).float()
        numerator = 2 * (predicted * truth).sum(dim=(-2, -1)) + 1.0
        denominator = predicted.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1)) + 1.0
        # No castiga una región ausente en ambos lados.
        present = truth.sum(dim=(-2, -1)) > 0
        if present.any():
            losses.append(1 - (numerator[present] / denominator[present]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def fragment_boundary_target(labels: Tensor) -> Tensor:
    """Bordes entre fragmentos distintos, sin marcar borde hueso-fondo."""
    target = torch.zeros_like(labels, dtype=torch.float32)
    vertical = (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
    horizontal = (labels[:, :, 1:] != labels[:, :, :-1]) & (labels[:, :, 1:] > 0) & (labels[:, :, :-1] > 0)
    target[:, 1:][vertical] = 1; target[:, :-1][vertical] = 1
    target[:, :, 1:][horizontal] = 1; target[:, :, :-1][horizontal] = 1
    return target


def multitask_loss(outputs: dict[str, Tensor], detection_target: Tensor,
                   labels: Tensor, weights: MultiTaskLossWeights | None = None,
                   detection_weights: DetectionLossWeights | None = None
                   ) -> tuple[Tensor, dict[str, float]]:
    """Pérdida compuesta con lambdas explícitas y registrables."""
    weights = weights or MultiTaskLossWeights()
    det, det_parts = detection_loss(outputs["detection"], detection_target,
                                    detection_weights)
    classification = F.binary_cross_entropy_with_logits(
        outputs["classification"], labels_to_presence(labels)
    )
    semantic_weights = torch.tensor([0.2, 1.0, 1.0, 1.0], device=labels.device)
    semantic_target = labels_to_semantic(labels)
    semantic = F.cross_entropy(outputs["semantic"], semantic_target,
                               weight=semantic_weights)
    semantic_dice = multiclass_foreground_dice_loss(outputs["semantic"], semantic_target)
    instance_weights = torch.ones(31, device=labels.device)
    instance_weights[0] = 0.1
    instance = F.cross_entropy(outputs["instance"], labels.long(),
                               weight=instance_weights)
    boundary_target = fragment_boundary_target(labels)
    boundary_probability = torch.sigmoid(outputs["boundary"].squeeze(1))
    boundary_bce = F.binary_cross_entropy_with_logits(
        outputs["boundary"].squeeze(1), boundary_target,
        pos_weight=torch.tensor(3.0, device=labels.device))
    boundary_numerator = 2 * (boundary_probability * boundary_target).sum(dim=(-2, -1)) + 1.0
    boundary_denominator = (boundary_probability.sum(dim=(-2, -1)) +
                            boundary_target.sum(dim=(-2, -1)) + 1.0)
    boundary = boundary_bce + (1 - boundary_numerator / boundary_denominator).mean()
    total = (weights.detection * det + weights.classification * classification +
             weights.semantic * semantic + weights.instance * instance)
    total = total + weights.semantic_dice * semantic_dice + weights.boundary * boundary
    parts = {
        "total": float(total.detach()),
        "detection": float(det.detach()),
        "classification": float(classification.detach()),
        "semantic": float(semantic.detach()),
        "instance": float(instance.detach()),
        "semantic_dice": float(semantic_dice.detach()),
        "boundary": float(boundary.detach()),
        "det_objectness": det_parts["objectness"],
        "det_box": det_parts["box"],
        "det_classification": det_parts["classification"],
    }
    return total, parts
