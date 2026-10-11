"""Arquitectura multitarea PENGWIN para Semana 10.

Un backbone propio con CBAM alimenta tres cabezas entrenadas desde cero:
clasificación anatómica, detección por grid y segmentación semántica/instancia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


INSTANCE_MODES = {"ids": 31, "primary": 7}
"""Canales de la salida de instancia según la formulación.

- ``ids``: 31 canales, un canal por ID PENGWIN 0..30 (formulación original).
- ``primary``: 7 canales, fondo + (primario, secundario) por región. Sigue la
  formulación primario/secundario del challenge PENGWIN 2024: el ID de un
  fragmento secundario es arbitrario entre pacientes (solo indica orden por
  tamaño), así que se pide a la red distinguir únicamente "fragmento principal"
  vs "resto" dentro de cada hueso, y los secundarios se separan después por
  componentes conectados (ver instance_postprocess.primary_secondary_to_instances).
"""


class SegmentationHead(nn.Module):
    """Decoder común con salida semántica y salida de instancia."""

    def __init__(self, in_channels: int, semantic_classes: int = 4,
                 instance_classes: int = 31, instance_mode: str = "ids",
                 deep_supervision: bool = False):
        super().__init__()
        if instance_mode not in INSTANCE_MODES:
            raise ValueError(f"instance_mode debe ser uno de {sorted(INSTANCE_MODES)}")
        self.instance_mode = instance_mode
        instance_classes = INSTANCE_MODES[instance_mode]
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
        self.centers = nn.Conv2d(c3, 3, 1)
        self.offsets = nn.Conv2d(c3, 2, 1)
        # Supervisión profunda (E5): salidas auxiliares en las tres primeras etapas
        # del decodificador (/8, /4, /2). Solo existen y se calculan si se pide.
        self.deep_supervision = deep_supervision
        if deep_supervision:
            stage_channels = (c1, c2, c3)
            self.aux_semantic = nn.ModuleList(
                [nn.Conv2d(c, semantic_classes, 1) for c in stage_channels])
            self.aux_instance = nn.ModuleList(
                [nn.Conv2d(c, instance_classes, 1) for c in stage_channels])

    def forward(self, features: Tensor, skips: list[Tensor] | None = None,
                return_aux: bool = False):
        decoded = features
        aux_semantic, aux_instance = [], []
        for index, block in enumerate(self.decoder):
            decoded = block(decoded)
            if skips is not None and index < 3:
                decoded = decoded + self.skip_projections[index](skips[-1 - index])
            if return_aux and self.deep_supervision and index < 3:
                aux_semantic.append(self.aux_semantic[index](decoded))
                aux_instance.append(self.aux_instance[index](decoded))
        outputs = (self.semantic(decoded), self.instance(decoded), self.boundary(decoded),
                   self.centers(decoded), torch.tanh(self.offsets(decoded)))
        if return_aux:
            return outputs, aux_semantic, aux_instance
        return outputs


class PengwinMultiTaskModel(nn.Module):
    """Backbone+CBAM compartido y tres cabezas requeridas por el proyecto."""

    def __init__(self, base_channels: int = 8, instance_mode: str = "ids", in_channels: int = 1,
                 use_cbam: bool = True, cbam_stages: int = 1, learnable_gamma: bool = False,
                 deep_supervision: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        self.backbone = FundidoraBackbone(in_channels=in_channels, base_channels=base_channels,
                                          use_cbam=use_cbam, cbam_stages=cbam_stages,
                                          learnable_gamma=learnable_gamma)
        channels = self.backbone.out_channels
        self.instance_mode = instance_mode
        self.classification_head = ClassificationHead(channels)
        self.detection_head = GridDetectionHead(channels)
        self.segmentation_head = SegmentationHead(channels, instance_mode=instance_mode,
                                                  deep_supervision=deep_supervision)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features, skips = self.backbone.forward_with_skips(x)
        use_aux = self.deep_supervision and self.training
        if use_aux:
            (semantic, instance, boundary, centers, offsets), aux_s, aux_i = \
                self.segmentation_head(features, skips, return_aux=True)
        else:
            semantic, instance, boundary, centers, offsets = self.segmentation_head(features, skips)
        extra = {"aux_semantic": aux_s, "aux_instance": aux_i} if use_aux else {}
        return {**extra,
            "classification": self.classification_head(features),
            "detection": self.detection_head(features),
            "semantic": semantic,
            "instance": instance,
            "boundary": boundary,
            "centers": centers,
            "offsets": offsets,
        }


def labels_to_semantic(labels: Tensor) -> Tensor:
    """IDs PENGWIN 0..30 → fondo/sacro/coxal izquierdo/coxal derecho."""
    semantic = torch.zeros_like(labels, dtype=torch.long)
    semantic[(labels >= 1) & (labels <= 10)] = 1
    semantic[(labels >= 11) & (labels <= 20)] = 2
    semantic[(labels >= 21) & (labels <= 30)] = 3
    return semantic


def labels_to_primary_secondary(labels: Tensor) -> Tensor:
    """IDs PENGWIN 0..30 → 7 clases: fondo + (primario, secundario) por región.

    0 fondo · 1 sacro primario · 2 sacro secundario · 3 coxal izq primario ·
    4 coxal izq secundario · 5 coxal der primario · 6 coxal der secundario.
    El primario es el ID 1/11/21, que la Semana 8 verificó como el fragmento
    de mayor volumen en 300/300 casos-región.
    """
    target = torch.zeros_like(labels, dtype=torch.long)
    for region, start in enumerate((1, 11, 21)):
        primary = labels == start
        secondary = (labels > start) & (labels <= start + 9)
        target[primary] = 2 * region + 1
        target[secondary] = 2 * region + 2
    return target


def flip_sagittal(images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Espejo izquierda↔derecha del corte axial con intercambio de coxales.

    Aumento de datos usado por los equipos del challenge PENGWIN 2024: un
    corte espejado sigue siendo una pelvis válida si los IDs del coxal
    izquierdo (11..20) pasan a ser del derecho (21..30) y viceversa. El sacro
    (1..10) no cambia. Duplica el conjunto de entrenamiento sin nuevos datos.
    """
    flipped_images = torch.flip(images, dims=(-1,))
    flipped_labels = torch.flip(labels, dims=(-1,)).clone()
    left = (flipped_labels >= 11) & (flipped_labels <= 20)
    right = (flipped_labels >= 21) & (flipped_labels <= 30)
    flipped_labels[left] += 10
    flipped_labels[right] -= 10
    return flipped_images, flipped_labels


def fracture_distance_weight(labels: Tensor, strength: float = 4.0,
                             sigma: float = 3.0) -> Tensor:
    """Mapa de pesos 1 + strength·exp(-d²/2σ²) con d = distancia (px) a una
    frontera entre fragmentos distintos del mismo hueso.

    Idea de FracSegNet (baseline del challenge): dar más importancia a los
    píxeles junto a la línea de fractura, que es donde la semántica fusiona
    fragmentos en contacto. Con strength=0 la pérdida vuelve a la original.
    """
    if strength <= 0:
        return torch.ones_like(labels, dtype=torch.float32)
    from scipy import ndimage as ndi
    boundary = fragment_boundary_target(labels).cpu().numpy() > 0
    weight = torch.ones(labels.shape, dtype=torch.float32)
    for bi in range(labels.shape[0]):
        if not boundary[bi].any():
            continue
        distance = ndi.distance_transform_edt(~boundary[bi])
        weight[bi] = 1.0 + strength * torch.from_numpy(
            np.exp(-(distance ** 2) / (2 * sigma ** 2))).float()
    return weight.to(labels.device)


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
    centers: float = 1.0
    offsets: float = 1.0
    instance_dice: float = 1.0
    fracture_weight: float = 0.0  # >0 activa el peso por distancia a la fractura
    secondary_weight: float = 1.0  # peso CE de las clases 'secundario' (modo primary)
    focal_gamma: float = 0.0       # >0: CE de instancia focal (E4)
    tversky_beta: float = 0.0      # >0: pérdida de instancia Tversky con peso beta a los falsos negativos (E4)
    deep_supervision: float = 0.0  # peso de las salidas auxiliares del decodificador (E5)


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


def fragment_geometry_targets(labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Heatmaps de centros y offsets normalizados hacia el centro de instancia."""
    batch, height, width = labels.shape
    centers = torch.zeros((batch, 3, height, width), device=labels.device)
    offsets = torch.zeros((batch, 2, height, width), device=labels.device)
    valid = labels > 0
    yy, xx = torch.meshgrid(torch.arange(height, device=labels.device),
                            torch.arange(width, device=labels.device), indexing="ij")
    for bi in range(batch):
        for fragment_id in torch.unique(labels[bi]).tolist():
            if fragment_id == 0:
                continue
            mask = labels[bi] == fragment_id
            cy = yy[mask].float().mean(); cx = xx[mask].float().mean()
            region = min((int(fragment_id) - 1) // 10, 2)
            iy, ix = int(torch.round(cy)), int(torch.round(cx))
            radius = 3
            y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
            x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
            distance2 = (yy[y0:y1, x0:x1] - cy).float().square() + (xx[y0:y1, x0:x1] - cx).float().square()
            centers[bi, region, y0:y1, x0:x1] = torch.maximum(
                centers[bi, region, y0:y1, x0:x1], torch.exp(-distance2 / 4.0))
            offsets[bi, 0][mask] = (cx - xx[mask]) / width
            offsets[bi, 1][mask] = (cy - yy[mask]) / height
    return centers, offsets, valid


def instance_dice_loss(logits: Tensor, labels: Tensor, tversky_beta: float = 0.0) -> Tensor:
    """Dice medio por ID de fragmento presente, excluyendo fondo.

    A diferencia de CE, cada fragmento visible contribuye por igual, incluso
    los conminutos pequeños que de otro modo quedan eclipsados por el hueso
    principal y el fondo.
    """
    probabilities = torch.softmax(logits, dim=1)
    losses = []
    for fragment_id in range(1, probabilities.shape[1]):
        truth = labels == fragment_id
        if not truth.any():
            continue
        predicted = probabilities[:, fragment_id]
        truth_float = truth.float()
        present = truth.sum(dim=(-2, -1)) > 0
        if tversky_beta > 0:
            # Tversky: alfa = 1 - beta pesa los falsos positivos y beta los falsos
            # negativos; con beta > 0.5 se castiga más omitir fragmento pequeño.
            tp = (predicted * truth_float).sum(dim=(-2, -1))
            fp = (predicted * (1 - truth_float)).sum(dim=(-2, -1))
            fn = ((1 - predicted) * truth_float).sum(dim=(-2, -1))
            score = (tp + 1.0) / (tp + (1 - tversky_beta) * fp + tversky_beta * fn + 1.0)
            losses.append(1 - score[present].mean())
            continue
        numerator = 2 * (predicted * truth_float).sum(dim=(-2, -1)) + 1.0
        denominator = predicted.sum(dim=(-2, -1)) + truth_float.sum(dim=(-2, -1)) + 1.0
        losses.append(1 - (numerator[present] / denominator[present]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def multitask_loss(outputs: dict[str, Tensor], detection_target: Tensor,
                   labels: Tensor, weights: MultiTaskLossWeights | None = None,
                   detection_weights: DetectionLossWeights | None = None,
                   fracture: Tensor | None = None
                   ) -> tuple[Tensor, dict[str, float]]:
    """Pérdida compuesta con lambdas explícitas y registrables.

    ``fracture`` permite pasar el mapa de peso por distancia a la fractura ya
    calculado (se precomputa una vez por corte en el entrenamiento; el mapa
    depende solo de las etiquetas).
    """
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
    if fracture is None:
        fracture = fracture_distance_weight(labels, weights.fracture_weight)
    if weights.fracture_weight > 0:
        # Reponderación por píxel de la CE semántica (FracSegNet).
        semantic = (F.cross_entropy(outputs["semantic"], semantic_target,
                                    weight=semantic_weights, reduction="none")
                    * fracture).sum() / fracture.sum()
    instance_channels = outputs["instance"].shape[1]
    if instance_channels == INSTANCE_MODES["primary"]:
        instance_target = labels_to_primary_secondary(labels)
    else:
        instance_target = labels.long()
    instance_weights = torch.ones(instance_channels, device=labels.device)
    instance_weights[0] = 0.1
    if instance_channels == INSTANCE_MODES["primary"]:
        # Los secundarios son ~24 % de los fragmentos pero una fracción mucho
        # menor de los píxeles; sin reponderar, la CE favorece predecir "primario".
        instance_weights[[2, 4, 6]] = weights.secondary_weight
    instance = F.cross_entropy(outputs["instance"], instance_target,
                               weight=instance_weights, reduction="none")
    if weights.focal_gamma > 0:
        # Focal (Lin et al.): (1 - p_t)^gamma baja el peso de los píxeles ya fáciles.
        plain = F.cross_entropy(outputs["instance"], instance_target, reduction="none")
        instance = instance * (1 - torch.exp(-plain)).clamp(min=0).pow(weights.focal_gamma)
    instance = (instance * fracture).sum() / fracture.sum()
    instance_dice = instance_dice_loss(outputs["instance"], instance_target,
                                       weights.tversky_beta)
    boundary_target = fragment_boundary_target(labels)
    boundary_probability = torch.sigmoid(outputs["boundary"].squeeze(1))
    boundary_bce = F.binary_cross_entropy_with_logits(
        outputs["boundary"].squeeze(1), boundary_target,
        pos_weight=torch.tensor(3.0, device=labels.device))
    boundary_numerator = 2 * (boundary_probability * boundary_target).sum(dim=(-2, -1)) + 1.0
    boundary_denominator = (boundary_probability.sum(dim=(-2, -1)) +
                            boundary_target.sum(dim=(-2, -1)) + 1.0)
    boundary = boundary_bce + (1 - boundary_numerator / boundary_denominator).mean()
    center_target, offset_target, offset_valid = fragment_geometry_targets(labels)
    center_probability = torch.sigmoid(outputs["centers"])
    center_weight = 1.0 + 20.0 * center_target
    centers = ((center_probability - center_target).square() * center_weight).mean()
    if offset_valid.any():
        offsets = F.smooth_l1_loss(outputs["offsets"].permute(0, 2, 3, 1)[offset_valid],
                                   offset_target.permute(0, 2, 3, 1)[offset_valid])
    else:
        # Un lote formado solo por cortes negativos no contiene fragmentos y,
        # por tanto, tampoco offsets válidos. La media de una selección vacía
        # es NaN; devolver un cero conectado al grafo mantiene el backward sano.
        offsets = outputs["offsets"].sum() * 0.0
    total = (weights.detection * det + weights.classification * classification +
             weights.semantic * semantic + weights.instance * instance)
    total = (total + weights.semantic_dice * semantic_dice + weights.boundary * boundary +
             weights.centers * centers + weights.offsets * offsets)
    total = total + weights.instance_dice * instance_dice
    deep = None
    if weights.deep_supervision > 0 and outputs.get("aux_semantic"):
        deep = 0.0
        size = labels.shape[-2:]
        for aux_s, aux_i in zip(outputs["aux_semantic"], outputs["aux_instance"]):
            aux_s = F.interpolate(aux_s, size=size, mode="bilinear", align_corners=False)
            aux_i = F.interpolate(aux_i, size=size, mode="bilinear", align_corners=False)
            deep = deep + F.cross_entropy(aux_s, semantic_target, weight=semantic_weights)
            deep = deep + F.cross_entropy(aux_i, instance_target, weight=instance_weights)
        deep = deep / len(outputs["aux_semantic"])
        total = total + weights.deep_supervision * deep
    parts = {
        "total": float(total.detach()),
        "detection": float(det.detach()),
        "classification": float(classification.detach()),
        "semantic": float(semantic.detach()),
        "instance": float(instance.detach()),
        "semantic_dice": float(semantic_dice.detach()),
        "boundary": float(boundary.detach()),
        "centers": float(centers.detach()),
        "offsets": float(offsets.detach()),
        "instance_dice": float(instance_dice.detach()),
        "deep_supervision": float(deep.detach()) if deep is not None else 0.0,
        "det_objectness": det_parts["objectness"],
        "det_box": det_parts["box"],
        "det_classification": det_parts["classification"],
    }
    return total, parts


# ---------------------------------------------------------------------------
# Segundo pase por región (Step 2 de FracSegNet, Liu et al. Front. Med. 2025):
# recortar la caja de cada hueso, enmascarar con su región anatómica y volver a
# segmentar fragmentos a mayor resolución efectiva. La misma red se entrena con
# cortes completos y con recortes, así el segundo pase no cambia de dominio.
# ---------------------------------------------------------------------------

def _region_box(mask: Tensor, margin: float) -> tuple[int, int, int, int] | None:
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(ys) == 0:
        return None
    height, width = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    my, mx = int((y1 - y0) * margin) + 2, int((x1 - x0) * margin) + 2
    return max(0, y0 - my), min(height, y1 + my), max(0, x0 - mx), min(width, x1 + mx)


def _dilate(mask: Tensor, radius: int) -> Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask.float()[None, None], kernel, stride=1, padding=radius)[0, 0] > 0


def region_crops(images: Tensor, labels: Tensor, semantic: Tensor | None = None,
                 margin: float = 0.10, dilate: int = 3, size: int | None = None,
                 min_pixels: int = 32) -> tuple[Tensor, Tensor, list[tuple[int, int, int, int, int, int]]]:
    """Recorta cada región anatómica presente y la reescala a ``size``.

    ``semantic`` (fondo/sacro/izq/der) define la región; si es None se deriva
    de ``labels`` (entrenamiento). Fuera de la región dilatada la imagen se
    pone a 0, como en FracSegNet. Devuelve imágenes [N,1,S,S], labels [N,S,S]
    con los IDs de la región y la lista (índice, región, y0, y1, x0, x1).
    """
    size = size or images.shape[-1]
    if semantic is None:
        semantic = labels_to_semantic(labels)
    crops, crop_labels, boxes = [], [], []
    for bi in range(images.shape[0]):
        for region in (1, 2, 3):
            mask = semantic[bi] == region
            if int(mask.sum()) < min_pixels:
                continue
            box = _region_box(mask, margin)
            if box is None:
                continue
            y0, y1, x0, x1 = box
            keep = _dilate(mask, dilate)[y0:y1, x0:x1]
            image = images[bi, :, y0:y1, x0:x1] * keep.to(images.dtype)
            label = labels[bi, y0:y1, x0:x1].clone()
            start = 10 * (region - 1) + 1
            label[(label < start) | (label > start + 9)] = 0
            crops.append(F.interpolate(image[None], size=(size, size), mode="bilinear",
                                       align_corners=False)[0])
            crop_labels.append(F.interpolate(label[None, None].float(), size=(size, size),
                                             mode="nearest")[0, 0].long())
            boxes.append((bi, region, y0, y1, x0, x1))
    if not crops:
        empty = images[:0]
        return empty, labels[:0], []
    return torch.stack(crops), torch.stack(crop_labels), boxes


def two_pass_instances(model: nn.Module, images: Tensor, semantic: Tensor,
                       margin: float = 0.10, dilate: int = 3) -> Tensor:
    """Mapa primario/secundario (7 clases) refinado con un segundo pase por región.

    Para cada región presente en ``semantic`` recorta, enmascara, reescala,
    vuelve a pasar por la red y pega de vuelta solo las clases de esa región.
    Donde no hay recorte se conserva el primer pase.
    """
    model.eval()
    with torch.inference_mode():
        first = model(images)["instance"].argmax(1)
        output = first.clone()
        size = images.shape[-1]
        dummy = torch.zeros_like(semantic)
        crops, _, boxes = region_crops(images, dummy, semantic, margin, dilate, size)
        if not boxes:
            return output
        preds = []
        for start in range(0, len(crops), 16):
            preds.append(model(crops[start:start + 16])["instance"].argmax(1))
        preds = torch.cat(preds)
        for pred, (bi, region, y0, y1, x0, x1) in zip(preds, boxes):
            local = F.interpolate(pred[None, None].float(), size=(y1 - y0, x1 - x0),
                                  mode="nearest")[0, 0].long()
            primary, secondary = 2 * region - 1, 2 * region
            inside = semantic[bi, y0:y1, x0:x1] == region
            view = output[bi, y0:y1, x0:x1]
            # Dentro de la región: lo que diga el segundo pase (si es de esta región).
            take = inside & ((local == primary) | (local == secondary))
            view[inside] = 0
            view[take] = local[take]
            # Píxeles de la región que el recorte no clasificó → primario (evita huecos).
            hole = inside & (view == 0)
            view[hole] = primary
    return output
