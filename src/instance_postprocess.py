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
