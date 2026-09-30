"""Convierte máscaras anatómicas en instancias de fragmentos por conectividad."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def semantic_to_instances(semantic: np.ndarray, boundary: np.ndarray | None = None,
                          boundary_threshold: float = 0.50, min_size: int = 12) -> np.ndarray:
    """Etiqueta componentes 8-conectados por región sin reutilizar IDs GT.

    El ID 1/11/21 se reserva para el componente de mayor área de cada región,
    que es la aproximación reproducible al macrofragmento en inferencia.
    """
    if semantic.ndim == 2:
        semantic = semantic[None]
    if boundary is not None and boundary.ndim == 2:
        boundary = boundary[None]
    output = np.zeros_like(semantic, dtype=np.uint8)
    starts = {1: 1, 2: 11, 3: 21}
    for index, image in enumerate(semantic):
        for category, start in starts.items():
            region = image == category
            if boundary is not None:
                region &= boundary[index] < boundary_threshold
            components, n = ndi.label(region, structure=np.ones((3, 3), dtype=np.uint8))
            areas = [(component, int((components == component).sum()))
                     for component in range(1, n + 1)]
            areas = [(component, area) for component, area in areas if area >= min_size]
            areas.sort(key=lambda row: row[1], reverse=True)
            for rank, (component, _) in enumerate(areas[:10]):
                output[index][components == component] = start + rank
    return output[0] if output.shape[0] == 1 else output


def geometry_to_instances(semantic: np.ndarray, centers: np.ndarray, offsets: np.ndarray,
                          center_threshold: float = 0.25, min_size: int = 12) -> np.ndarray:
    """Agrupa píxeles por centro predicho dentro de cada macro-región."""
    output = np.zeros_like(semantic, dtype=np.uint8)
    starts = (1, 11, 21)
    for bi, image in enumerate(semantic):
        height, width = image.shape
        yy, xx = np.indices((height, width))
        for region, start in enumerate(starts):
            mask = image == region + 1
            if not mask.any():
                continue
            heat = centers[bi, region]
            peaks = (heat == ndi.maximum_filter(heat, size=9)) & (heat >= center_threshold) & mask
            py, px = np.where(peaks)
            if not len(px):
                fallback = semantic_to_instances(image, min_size=min_size)
                region_mask = (fallback >= start) & (fallback <= start + 9)
                output[bi][region_mask] = fallback[region_mask]
                continue
            order = np.argsort(heat[py, px])[::-1][:10]
            py, px = py[order], px[order]
            ys, xs = np.where(mask)
            voted_x = xs + offsets[bi, 0, ys, xs] * width
            voted_y = ys + offsets[bi, 1, ys, xs] * height
            distances = ((voted_x[:, None] - px[None]) ** 2 +
                         (voted_y[:, None] - py[None]) ** 2)
            assignments = distances.argmin(axis=1)
            areas = [(index, int((assignments == index).sum())) for index in range(len(px))]
            areas = [(index, area) for index, area in areas if area >= min_size]
            areas.sort(key=lambda row: row[1], reverse=True)
            for rank, (index, _) in enumerate(areas[:10]):
                selected = assignments == index
                output[bi, ys[selected], xs[selected]] = start + rank
    return output
