"""Inferencia sobre un volumen CT sin ground truth (Semana 11, para los visualizadores).

``evaluate_volumes.evaluate_case`` exige las etiquetas porque calcula métricas. La demo no las
tiene: parte de un volumen crudo en HU y devuelve, por corte, lo que el dashboard dibuja.

Pasos, los mismos del entrenamiento (``pengwin_io`` es la única fuente de preprocesamiento):

1. ventana ósea + gamma 1.5 y resize a ``size`` × ``size`` por corte (``preprocess_volume``);
2. modelo de tres cabezas (opcionalmente con TTA de espejo);
3. detección: cabeza grid + NMS propio, una caja cruda por región y la caja refinada con la
   semántica (igual que ``evaluate_volumes``);
4. mapa semántico (4 clases) y mapa primario/secundario (7 clases) por corte.

El post-proceso 3D y las distancias en mm están en ``demo_export``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_cache import from_uint8, to_uint8
from pengwin_io import GAMMA, resize_slice, window_hu
from pengwin_metrics import refine_boxes_with_semantic
from week9_detection import decode_predictions

REGION_NAMES = ("sacro", "coxal_izquierdo", "coxal_derecho")


def preprocess_volume(image_hu: np.ndarray, size: int = 384, gamma: float = GAMMA) -> np.ndarray:
    """Volumen HU crudo [z,y,x] → uint8 [z,size,size], igual que ``build_case_cache`` (modo stretch)."""
    out = np.empty((len(image_hu), size, size), dtype=np.uint8)
    for z in range(len(image_hu)):
        out[z] = to_uint8(resize_slice(window_hu(image_hu[z], gamma=gamma), size))
    return out


def cache_spacing_xyz(spacing_xyz: tuple[float, float, float], shape_zyx: tuple[int, int, int],
                      size: int) -> tuple[float, float, float]:
    """Spacing (x,y,z) en mm del volumen estirado a size×size por corte.

    Cada píxel del caché cubre más o menos milímetros que el original según la forma del corte.
    Misma fórmula que ``volume_postprocess.cache_spacing_xyz``, pero desde el header.
    """
    sx, sy, sz = (float(v) for v in spacing_xyz)
    return (sx * shape_zyx[2] / size, sy * shape_zyx[1] / size, sz)


def _box_list(rows: list[dict]) -> list[dict]:
    """Cajas serializables: {region_id, region, score, box:[x0,y0,x1,y1] normalizado}."""
    out = []
    for row in rows:
        box = row["box"]
        box = [float(v) for v in (box.tolist() if hasattr(box, "tolist") else box)]
        cls = int(row["class_id"])
        out.append({"region_id": cls, "region": REGION_NAMES[cls],
                    "score": float(row["score"]), "box": box})
    return out


def _context_stack(images_u8: np.ndarray, start: int, stop: int, context: int) -> np.ndarray:
    depth = len(images_u8)
    z = np.arange(start, stop)
    return np.stack([images_u8[np.clip(z + d, 0, depth - 1)] for d in range(-context, context + 1)], axis=1)


def predict_volume(model: torch.nn.Module, images_u8: np.ndarray, device: torch.device,
                   batch_size: int = 8, confidence: float = 0.10,
                   progress: Callable[[int, int], None] | None = None) -> dict:
    """Infiere todos los cortes. Devuelve arrays [z,S,S] y listas por corte.

    - ``semantic`` uint8: 0 fondo, 1 sacro, 2 coxal izquierdo, 3 coxal derecho.
    - ``primary_secondary`` uint8: 0 fondo; 2*r+1 primario y 2*r+2 secundario de la región r.
    - ``classification`` float32 [z,3]: probabilidad de presencia de cada región.
    - ``boxes_raw`` / ``boxes_refined``: por corte, las cajas tras el NMS propio (una por región
      la cruda; la refinada con la semántica es la que evalúa el proyecto).
    - ``seconds``: tiempo total de las pasadas del modelo y la decodificación.
    """
    if getattr(model, "instance_mode", "primary") != "primary":
        raise ValueError("La demo necesita un modelo con instancia 'primary' (mapa de 7 clases).")
    model.eval()
    depth = len(images_u8)
    context = (int(getattr(model, "in_channels", 1)) - 1) // 2
    semantic, primary_secondary, cls_prob = [], [], []
    boxes_raw: list[list[dict]] = []
    boxes_refined: list[list[dict]] = []
    started = time.perf_counter()
    for start in range(0, depth, batch_size):
        stop = min(start + batch_size, depth)
        if context > 0:
            x = torch.from_numpy(from_uint8(_context_stack(images_u8, start, stop, context)))
        else:
            x = torch.from_numpy(from_uint8(images_u8[start:stop]))[:, None]
        with torch.inference_mode():
            output = model(x.to(device))
        sem = output["semantic"].argmax(1).cpu().numpy().astype(np.uint8)
        semantic.append(sem)
        primary_secondary.append(output["instance"].argmax(1).cpu().numpy().astype(np.uint8))
        cls_prob.append(torch.sigmoid(output["classification"]).cpu().numpy().astype(np.float32))
        detection = output["detection"].cpu()
        all_boxes = decode_predictions(detection, confidence=confidence)
        raw_one = decode_predictions(detection, confidence=confidence, max_por_clase=1)
        refined = refine_boxes_with_semantic(all_boxes, sem)
        boxes_raw.extend(_box_list(rows) for rows in raw_one)
        boxes_refined.extend(_box_list(rows) for rows in refined)
        if progress:
            progress(stop, depth)
    return {"semantic": np.concatenate(semantic), "primary_secondary": np.concatenate(primary_secondary),
            "classification": np.concatenate(cls_prob), "boxes_raw": boxes_raw,
            "boxes_refined": boxes_refined, "seconds": time.perf_counter() - started}


def load_pipeline(checkpoint: str | Path, device: torch.device | None = None, tta: bool = False):
    """Carga el checkpoint con el mismo código que la evaluación. Devuelve (modelo, device, info)."""
    from evaluate_volumes import load_model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model([str(checkpoint)], device, tta)
    raw = torch.load(checkpoint, map_location="cpu")
    info = {"checkpoint": Path(checkpoint).name, "base_channels": int(raw["base_channels"]),
            "input_size": int(raw.get("input_size", 384)), "ema": raw.get("ema"),
            "parameters": int(sum(p.numel() for p in model.parameters())), "tta": bool(tta),
            # True: el modelo espera cortes en orientación canónica (orientacion.py); quien lo llama
            # endereza el volumen antes. predict_volume no gira nada.
            "canonical": bool(getattr(model, "canonical", False))}
    return model, device, info


def predict_slice(model: torch.nn.Module, device: torch.device, image_u8: np.ndarray,
                  confidence: float = 0.10) -> dict:
    """Inferencia de UN corte uint8 [S,S] con su latencia en ms (para el botón de demo en vivo).

    Mide solo modelo + decodificación, sin dibujar. Calienta con una pasada previa si es la primera.
    """
    started = time.perf_counter()
    result = predict_volume(model, image_u8[None], device, batch_size=1, confidence=confidence)
    result["latency_ms"] = (time.perf_counter() - started) * 1000.0
    return result
