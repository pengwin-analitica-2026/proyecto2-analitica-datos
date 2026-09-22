"""Inferencia corte a corte del pipeline multitarea de Semana 10.

La secuencia es deliberadamente explícita para el informe: el detector produce
regiones y NMS elimina duplicados; después se restringe la salida de instancia
al bounding box y se asigna cada ID de fragmento a su macro-región.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from week9_detection import decode_predictions
from week10_multitask import PengwinMultiTaskModel


REGION_NAMES = {0: "sacro", 1: "coxal_izquierdo", 2: "coxal_derecho"}


def region_for_fragment(fragment_id: int) -> int | None:
    if 1 <= fragment_id <= 10:
        return 0
    if 11 <= fragment_id <= 20:
        return 1
    if 21 <= fragment_id <= 30:
        return 2
    return None


def _box_to_pixels(box: list[float], height: int, width: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (max(0, min(width, int(np.floor(x0 * width)))),
            max(0, min(height, int(np.floor(y0 * height)))),
            max(0, min(width, int(np.ceil(x1 * width)))),
            max(0, min(height, int(np.ceil(y1 * height)))))


def infer_slice(model: torch.nn.Module, image: Tensor, *, confidence: float = 0.20,
                iou_threshold: float = 0.50) -> dict[str, object]:
    """Ejecuta una imagen [1,1,H,W] y devuelve resultados serializables."""
    if image.ndim != 4:
        raise ValueError("image debe tener forma [B,1,H,W]")
    model.eval()
    with torch.inference_mode():
        outputs = model(image)
        detections = decode_predictions(outputs["detection"], confidence, iou_threshold)[0]
        semantic = outputs["semantic"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        instance = outputs["instance"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    height, width = semantic.shape
    fragments = np.zeros((height, width), dtype=np.uint8)
    rows: list[dict[str, object]] = []
    for detection in detections:
        region_id = int(detection["class_id"])
        box = detection["box"]
        x0, y0, x1, y1 = _box_to_pixels(box, height, width)
        if x1 <= x0 or y1 <= y0:
            continue
        candidate = instance[y0:y1, x0:x1]
        candidate_region = np.zeros_like(candidate, dtype=bool)
        for fragment_id in np.unique(candidate):
            if region_for_fragment(int(fragment_id)) == region_id:
                candidate_region |= candidate == fragment_id
        # Enforce the anatomical semantic prediction inside the detected region.
        candidate_region &= semantic[y0:y1, x0:x1] == region_id + 1
        fragment_view = fragments[y0:y1, x0:x1]
        fragment_view[candidate_region] = candidate[candidate_region]
        fragment_ids = sorted(int(v) for v in np.unique(candidate[candidate_region]) if v > 0)
        rows.append({
            "region_id": region_id,
            "region": REGION_NAMES.get(region_id, "desconocida"),
            "score": float(detection["score"]),
            "box": [float(v) for v in box],
            "fragment_ids": fragment_ids,
        })
    return {"detections": rows, "semantic": semantic, "fragments": fragments}


def load_checkpoint(path: str | Path, device: str | None = None) -> tuple[torch.nn.Module, torch.device]:
    """Carga un checkpoint guardado por train_week10_overfit."""
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(path, map_location=target)
    model = PengwinMultiTaskModel(base_channels=int(checkpoint.get("base_channels", 4)))
    model.load_state_dict(checkpoint["model"])
    return model.to(target), target

