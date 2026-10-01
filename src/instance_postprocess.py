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


def primary_secondary_to_instances(primary_secondary: np.ndarray,
                                   semantic: np.ndarray | None = None,
                                   min_size: int = 12, fill_holes: bool = False,
                                   absorb_small: int = 0) -> np.ndarray:
    """Mapa de 7 clases (fondo + primario/secundario por región) → IDs PENGWIN.

    Por región: el ID 1/11/21 es el componente conectado más grande de la
    clase "primario"; los demás componentes primarios pequeños se tratan como
    secundarios. Cada componente 8-conectado de la clase "secundario" recibe
    su propio ID (start+1, start+2, ...) ordenado por área. Si ``semantic`` se
    pasa, la región se restringe a la predicción anatómica para que instancia
    y semántica no se contradigan (mismo criterio que inference_week10).

    ``fill_holes`` rellena huecos internos de cada instancia; ``absorb_small``
    fusiona con su vecino más cercano de la misma región los componentes
    secundarios menores a ese número de píxeles en vez de descartarlos.
    Ambos se calibran sobre validación con calibrate_primary_postprocess.py.
    """
    if primary_secondary.ndim == 2:
        primary_secondary = primary_secondary[None]
    if semantic is not None and semantic.ndim == 2:
        semantic = semantic[None]
    output = np.zeros_like(primary_secondary, dtype=np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8)
    for index, image in enumerate(primary_secondary):
        for region, start in enumerate((1, 11, 21)):
            primary = image == 2 * region + 1
            secondary = image == 2 * region + 2
            if semantic is not None:
                anatomical = semantic[index] == region + 1
                primary &= anatomical
                secondary &= anatomical
            components, n = ndi.label(primary, structure=structure)
            if n > 0:
                areas = ndi.sum(primary, components, index=range(1, n + 1))
                largest = int(np.argmax(areas)) + 1
                output[index][components == largest] = start
                secondary |= primary & (components != largest)
            components, n = ndi.label(secondary, structure=structure)
            areas = [(component, int((components == component).sum()))
                     for component in range(1, n + 1)]
            areas = [(component, area) for component, area in areas if area >= min_size]
            areas.sort(key=lambda row: row[1], reverse=True)
            if n > 0 and not (output[index] == start).any() and areas:
                # Sin primario predicho: el secundario mayor asume el rol.
                component, _ = areas.pop(0)
                output[index][components == component] = start
            for rank, (component, _) in enumerate(areas[:9]):
                output[index][components == component] = start + 1 + rank
            if absorb_small > 0:
                region_ids = [value for value in np.unique(output[index])
                              if start <= value <= start + 9]
                for value in region_ids:
                    mask = output[index] == value
                    if value == start or mask.sum() >= absorb_small or len(region_ids) < 2:
                        continue
                    others = np.isin(output[index], [v for v in region_ids if v != value])
                    _, nearest = ndi.distance_transform_edt(~others, return_indices=True)
                    output[index][mask] = output[index][nearest[0][mask], nearest[1][mask]]
            if fill_holes:
                for value in np.unique(output[index]):
                    if start <= value <= start + 9:
                        filled = ndi.binary_fill_holes(output[index] == value)
                        output[index][filled & (output[index] == 0)] = value
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
