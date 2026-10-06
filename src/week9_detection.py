"""Semana 9: backbone propio, CBAM y deteccion 2D por grid.

La cabeza predice por celda:
    objectness, (cx, cy, w, h) normalizados y 3 clases anatómicas.

No usa YOLO, torchvision ni otro framework de detección. La salida es
compatible con el pipeline corte-a-corte que se construirá en semanas 10-11.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    use_cbam=False sustituye CBAM por nn.Identity y deja el resto de la red
    intacta. Es la rama de control de la ablación con y sin atención que pide
    el documento del proyecto; no es una configuración de entrega.

    cbam_stages=3 agrega CBAM también tras los bloques 2 y 3 (mapas /4 y /8),
    además del final (/16). learnable_gamma=True mezcla cada módulo con su
    entrada mediante un escalar aprendible por etapa,

        salida = x + gamma * (CBAM(x) - x),   gamma inicial = 0,

    para medir cuánto aporta la atención en cada capa (asesoría semana 9):
    gamma = 0 es "sin atención", gamma = 1 es CBAM completo. Con los valores
    por defecto (1 etapa, sin gamma) el modelo y sus nombres de parámetros son
    idénticos a los de antes, así que los checkpoints viejos cargan igual.
    """

    ETAPAS = {1: "etapa_4", 2: "etapa_8", 3: "etapa_16"}   # índice de bloque -> nombre

    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 use_cbam: bool = True, cbam_stages: int = 1,
                 learnable_gamma: bool = False):
        super().__init__()
        if cbam_stages not in (1, 3):
            raise ValueError("cbam_stages debe ser 1 o 3")
        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(in_channels, c, 2),
            ConvBlock(c, c * 2, 2),
            ConvBlock(c * 2, c * 4, 2),
            ConvBlock(c * 4, c * 8, 2),
        )
        self.use_cbam = use_cbam
        self.cbam_stages = cbam_stages if use_cbam else 0
        self.learnable_gamma = bool(learnable_gamma and use_cbam)
        self.cbam = CBAM(c * 8) if use_cbam else nn.Identity()
        canales = {1: c * 2, 2: c * 4}
        self.cbam_intermedios = nn.ModuleDict(
            {str(i): CBAM(canales[i]) for i in (1, 2)} if use_cbam and cbam_stages == 3 else {})
        if self.learnable_gamma:
            activos = [3] + [int(k) for k in self.cbam_intermedios]
            self.gammas = nn.ParameterDict({self.ETAPAS[i]: nn.Parameter(torch.zeros(1))
                                            for i in sorted(activos)})
        self.out_channels = c * 8

    def _atencion(self, indice: int, modulo: nn.Module, x: Tensor) -> Tensor:
        y = modulo(x)
        if self.learnable_gamma:
            gamma = self.gammas[self.ETAPAS[indice]]
            return x + gamma * (y - x)
        return y

    def _pasar(self, x: Tensor, skips: list[Tensor] | None = None) -> Tensor:
        for i, block in enumerate(self.features):
            x = block(x)
            if str(i) in self.cbam_intermedios:
                x = self._atencion(i, self.cbam_intermedios[str(i)], x)
            if skips is not None:
                skips.append(x)
        return self._atencion(3, self.cbam, x) if self.use_cbam else self.cbam(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._pasar(x)

    def forward_with_skips(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Retorna bottleneck con CBAM y mapas /2, /4, /8 para segmentación."""
        skips: list[Tensor] = []
        out = self._pasar(x, skips)
        return out, skips[:-1]

    def valores_gamma(self) -> dict[str, float]:
        """gamma aprendido por etapa (vacío si learnable_gamma=False)."""
        if not self.learnable_gamma:
            return {}
        return {k: float(v.detach()) for k, v in self.gammas.items()}


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
    """Modelo Semana 9: backbone + CBAM + cabeza de detección.

    use_cbam=False produce el modelo de control de la ablación: mismo backbone
    y misma cabeza, sin atención.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32,
                 use_cbam: bool = True, cbam_stages: int = 1,
                 learnable_gamma: bool = False):
        super().__init__()
        self.backbone = FundidoraBackbone(base_channels=base_channels, use_cbam=use_cbam,
                                          cbam_stages=cbam_stages,
                                          learnable_gamma=learnable_gamma)
        self.detection_head = GridDetectionHead(self.backbone.out_channels, num_classes)
        self.num_classes = num_classes
        self.use_cbam = use_cbam

    def forward(self, x: Tensor) -> Tensor:
        return self.detection_head(self.backbone(x))


def detection_model_from_checkpoint(checkpoint: dict) -> "DetectionModel":
    """Reconstruye el modelo con la configuración guardada y carga sus pesos.

    Los checkpoints anteriores (sin cbam_stages ni learnable_gamma) cargan
    como CBAM de una etapa sin gamma, que es lo que eran.
    """
    model = DetectionModel(base_channels=checkpoint["base_channels"],
                           use_cbam=checkpoint.get("use_cbam", True),
                           cbam_stages=checkpoint.get("cbam_stages", 1),
                           learnable_gamma=checkpoint.get("learnable_gamma", False))
    model.load_state_dict(checkpoint["model"])
    return model


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


def _celda_libre(ocupadas: set, gi: int, gj: int, gh: int, gw: int,
                 radio_max: int = 2) -> tuple[int, int] | None:
    """Celda libre más cercana a (gi, gj), buscando en anillos crecientes.

    Devuelve None si no hay ninguna dentro de radio_max. Se limita el radio a
    propósito: reubicar más lejos desplazaría demasiado el centro de la caja
    (ver labels_to_grid) y sería peor que no aprenderla.
    """
    if (gi, gj) not in ocupadas:
        return gi, gj
    for radio in range(1, radio_max + 1):
        candidatas = []
        for di in range(-radio, radio + 1):
            for dj in range(-radio, radio + 1):
                if max(abs(di), abs(dj)) != radio:      # solo el borde del anillo
                    continue
                ni, nj = gi + di, gj + dj
                if 0 <= ni < gh and 0 <= nj < gw and (ni, nj) not in ocupadas:
                    candidatas.append((di * di + dj * dj, ni, nj))
        if candidatas:
            _, ni, nj = min(candidatas)
            return ni, nj
    return None


def labels_to_grid(label: Tensor, num_classes: int = NUM_CLASSES,
                   stats: dict | None = None) -> Tensor:
    """Convierte máscaras [B,H,W] en targets [B,h,w,5+C].

    Las cajas se calculan por región anatómica. Si una región aparece en el
    corte, se usa una sola caja que cubre todos sus fragmentos visibles.

    Colisiones: dos regiones distintas pueden tener su centro dentro de la
    misma celda del grid (celdas de 16x16 px). La versión anterior conservaba
    solo la caja de mayor área y descartaba la otra en silencio, de modo que
    el modelo aprendía a NO predecir esa región mientras boxes_from_label sí
    la contaba al evaluar: el IoU bajaba por un defecto de codificación, no
    por el modelo. Ahora la región desplazada se reubica en la celda libre
    más cercana.

    Coste de la reubicación: tx/ty se guardan relativos a su celda y se
    decodifican con sigmoid, que solo cubre [0,1]. Al mover la caja de celda,
    su centro real cae fuera de ese rango, así que tx/ty se recortan al borde.
    El error máximo es de una celda (16 px sobre 256, ~6%) y solo afecta al
    centro; el ancho y el alto quedan exactos. Es preferible a perder la
    región entera, que es un error del 100%.

    stats: diccionario opcional; si se pasa, se llena con el número de
    regiones reubicadas y descartadas, para poder reportarlo en el informe en
    vez de que la pérdida ocurra en silencio.
    """
    if label.ndim != 3:
        raise ValueError(f"label debe ser [B,H,W], recibido {tuple(label.shape)}")
    b, height, width = label.shape
    gh, gw = height // 16, width // 16
    target = torch.zeros((b, gh, gw, 5 + num_classes), dtype=torch.float32, device=label.device)
    reubicadas = descartadas = 0
    for bi in range(b):
        ocupadas: set = set()
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

            destino = _celda_libre(ocupadas, gi, gj, gh, gw)
            if destino is None:
                # Sin celda libre cerca: no hay dónde codificarla sin mover
                # demasiado el centro. Se descarta, pero queda contabilizado.
                descartadas += 1
                continue
            if destino != (gi, gj):
                reubicadas += 1
            gi, gj = destino
            ocupadas.add((gi, gj))

            target[bi, gi, gj, 0] = 1
            # cx/cy se guardan relativos a la celda, igual que los logits
            # decodificados con sigmoid. w/h permanecen normalizados a la imagen.
            tx = (cx * gw - gj).clamp(0.0, 1.0)
            ty = (cy * gh - gi).clamp(0.0, 1.0)
            target[bi, gi, gj, 1:5] = torch.stack([tx, ty, bw, bh])
            target[bi, gi, gj, 5:] = 0
            target[bi, gi, gj, 5 + cls] = 1
    if stats is not None:
        stats["reubicadas"] = stats.get("reubicadas", 0) + reubicadas
        stats["descartadas"] = stats.get("descartadas", 0) + descartadas
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
                       iou_threshold: float = 0.5,
                       max_por_clase: int | None = None) -> list[list[dict[str, object]]]:
    """Decodifica la salida y aplica NMS independiente por clase.

    Es solo inferencia: se desconecta del grafo para no arrastrar gradientes
    (y evitar el aviso de PyTorch al convertir tensores con grad a float).

    max_por_clase: tope de cajas por clase y por corte tras el NMS (se quedan
    las de mayor puntaje). El NMS solo elimina cajas que se SOLAPAN (IoU >
    iou_threshold); dos cajas de la misma clase que cubren mitades distintas
    del coxal se solapan poco y sobreviven las dos, sin importar que una tenga
    0.95 y la otra 0.45. Como cada region (sacro, coxal izq., coxal der.)
    aparece una sola vez por corte y su caja real es una sola
    (boxes_from_label), con max_por_clase=1 toda caja extra se descarta.
    None conserva el comportamiento anterior (todas las que pasan el NMS).
    """
    if max_por_clase is not None and max_por_clase < 1:
        raise ValueError("max_por_clase debe ser >= 1 o None")
    if pred.ndim != 4:
        raise ValueError("pred debe tener forma [B,grid_y,grid_x,5+C]")
    pred = pred.detach()
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
            keep = nms(boxes, scores, iou_threshold).tolist()  # ya ordenado por puntaje
            if max_por_clase is not None:
                keep = keep[:max_por_clase]
            for idx in keep:
                item = candidates[idx].copy()
                item["box"] = boxes[idx].detach().cpu().tolist()
                item["score"] = float(scores[idx].detach())
                final.append(item)
        result.append(sorted(final, key=lambda r: float(r["score"]), reverse=True))
    return result


def match_detections(predicciones: list[dict[str, object]],
                     verdad: list[dict[str, object]],
                     iou_threshold: float = 0.5) -> dict[str, float]:
    """Empareja predicciones con cajas reales de un corte y cuenta aciertos.

    Motivación: la métrica anterior tomaba, por cada caja real, la predicción
    que mejor le calzara, y no miraba las demás. Un modelo que dispare veinte
    cajas por corte mejoraba su número sin ser mejor, porque los falsos
    positivos no costaban nada. Aquí el emparejamiento es uno a uno y en orden
    de confianza, así que toda predicción que sobre cuenta como falso positivo.

    Emparejamiento voraz: se recorren las predicciones de mayor a menor
    puntaje y cada una toma la caja real libre de su misma clase con mayor
    IoU, siempre que supere iou_threshold. Es el criterio estándar de
    detección y es el que hará falta para el mAP de la semana 10.

    Devuelve verdaderos positivos, falsos positivos, falsos negativos, y la
    suma del mejor IoU por caja real (iou_suma / n_gt da el IoU medio,
    contando como 0 las regiones que el modelo no detectó).
    """
    tp = fp = 0
    libres = list(range(len(verdad)))
    mejor_iou = [0.0] * len(verdad)

    for pred in sorted(predicciones, key=lambda p: float(p["score"]), reverse=True):
        candidatas = [k for k in libres if verdad[k]["class_id"] == pred["class_id"]]
        # El mejor IoU por caja real se mide sobre TODAS las predicciones de su
        # clase, estén libres o no; es la continuidad con la métrica anterior.
        for k, gt in enumerate(verdad):
            if gt["class_id"] != pred["class_id"]:
                continue
            iou = float(box_iou_xyxy(torch.tensor([gt["box"]]),
                                     torch.tensor([pred["box"]]))[0, 0])
            mejor_iou[k] = max(mejor_iou[k], iou)
        if not candidatas:
            fp += 1
            continue
        ious = [float(box_iou_xyxy(torch.tensor([verdad[k]["box"]]),
                                   torch.tensor([pred["box"]]))[0, 0]) for k in candidatas]
        mejor = int(max(range(len(candidatas)), key=lambda i: ious[i]))
        if ious[mejor] >= iou_threshold:
            tp += 1
            libres.remove(candidatas[mejor])
        else:
            fp += 1

    return {"tp": float(tp), "fp": float(fp), "fn": float(len(libres)),
            "n_gt": float(len(verdad)), "n_pred": float(len(predicciones)),
            "iou_suma": float(sum(mejor_iou))}


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