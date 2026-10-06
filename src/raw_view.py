"""Visualizador 1 (volumen crudo): MIP e isosuperficie del hueso, sin modelo.

El enunciado (sección 3.4) pide "una vista de proyección de máxima intensidad (MIP), mostrando
únicamente el hueso" con "umbral de intensidad en unidades Hounsfield sobre el volumen sin
procesar". Aquí se entregan las dos vistas:

- ``compute_mips``: MIP en los tres ejes. Por cada línea que atraviesa el volumen se toma el vóxel
  de mayor HU entre los que pasan la máscara ósea. Se remuestrea a ``target_mm`` por píxel para
  que cada eje salga con su proporción física.
- ``bone_mesh``: isosuperficie (marching cubes) de la misma máscara, con tope de triángulos para
  que abra en un celular.

La máscara es ``pengwin_io.clean_bone_mask`` (umbral 250 HU sobre el volumen sin procesar más la
limpieza de camilla y cables de la Semana 8). No interviene ningún modelo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pengwin_io import BONE_HU_THRESHOLD, clean_bone_mask

MIP_HU_MAX = 1500.0   # HU que se muestra como blanco; por encima se satura (cortical densa, metal)


def bone_mask_for(image_hu: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = (float(v) for v in spacing_xyz)
    return clean_bone_mask(image_hu, spacing_zyx=(sz, sy, sx))


def compute_mips(image_hu: np.ndarray, mask: np.ndarray, spacing_xyz: tuple[float, float, float],
                 target_mm: float = 1.0) -> dict[str, np.ndarray]:
    """MIP axial, coronal y sagital (uint8). Fuera de la máscara el valor es 0 (negro)."""
    sx, sy, sz = (float(v) for v in spacing_xyz)
    span = MIP_HU_MAX - BONE_HU_THRESHOLD
    scaled = np.clip((image_hu.astype(np.float32) - BONE_HU_THRESHOLD) / span, 0.0, 1.0)
    scaled[~mask] = 0.0
    views = {"axial": (scaled.max(axis=0), sy, sx),        # (y, x)
             "coronal": (scaled.max(axis=1), sz, sx),      # (z, x)
             "sagital": (scaled.max(axis=2), sz, sy)}      # (z, y)
    out = {}
    for name, (image, row_mm, col_mm) in views.items():
        shape = (max(int(round(image.shape[0] * row_mm / target_mm)), 1),
                 max(int(round(image.shape[1] * col_mm / target_mm)), 1))
        image = resize(image, shape, order=1, preserve_range=True, anti_aliasing=True)
        out[name] = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    # Coronal y sagital con la cabeza arriba. En los casos de PENGWIN revisados el índice z crece
    # hacia la cabeza, así que la fila 0 (pies) se voltea al final. Si un caso viniera invertido,
    # se vería con los fémures arriba; el header trae la orientación si hiciera falta corregirlo.
    out["coronal"] = out["coronal"][::-1]
    out["sagital"] = out["sagital"][::-1]
    return out


def mesh_from_mask(mask: np.ndarray, spacing_zyx: tuple[float, float, float], factor: int,
                   smooth: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes sobre una máscara reducida ``factor`` veces. Vértices (z,y,x) en mm."""
    if factor > 1:
        field = ndi.uniform_filter(mask.astype(np.float32), size=factor)[::factor, ::factor, ::factor]
        spacing_zyx = tuple(s * factor for s in spacing_zyx)
    else:
        field = ndi.gaussian_filter(mask.astype(np.float32), 0.8) if smooth else mask.astype(np.float32)
    field = np.pad(field, 1)
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5, spacing=spacing_zyx)
    verts = verts - np.asarray(spacing_zyx, dtype=np.float64)   # deshace el padding de 1 vóxel
    return verts.astype(np.float32), faces.astype(np.int32)


def bone_mesh(mask: np.ndarray, spacing_xyz: tuple[float, float, float],
              max_faces: int = 150_000) -> dict:
    """Isosuperficie del hueso con tope de triángulos. Sube el factor de reducción hasta cumplirlo."""
    sx, sy, sz = (float(v) for v in spacing_xyz)
    spacing_zyx = (sz, sy, sx)
    if not mask.any():
        raise ValueError("La máscara ósea está vacía")
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - 2, 0)
    hi = coords.max(axis=0) + 3
    crop = mask[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
    for factor in (2, 3, 4, 5, 6, 8):
        verts, faces = mesh_from_mask(crop, spacing_zyx, factor)
        if len(faces) <= max_faces:
            break
    verts = verts + (lo * np.asarray(spacing_zyx)).astype(np.float32)
    return {"verts_zyx": verts, "faces": faces, "factor": factor}
