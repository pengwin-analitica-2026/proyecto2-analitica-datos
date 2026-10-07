"""Orientación canónica de los volúmenes PENGWIN.

34 de los 100 .mha traen ``TransformMatrix = -1 0 0 0 -1 0 0 0 1`` (orientación LPI) y los otros
66 la identidad (RAI). ``SimpleITK.GetArrayFromImage`` devuelve la matriz tal como está guardada,
sin aplicar esa dirección, así que un tercio de los casos entra al modelo girado 180 grados en el
plano axial: el sacro queda arriba y el coxal izquierdo del paciente a la izquierda de la imagen,
al revés que en el resto. La red tenía que adivinar la orientación de cada corte para decidir
izquierda y derecha.

``to_canonical`` deshace ese giro (invierte filas y columnas) para que todos los casos queden
como los RAI: sacro abajo, coxal izquierdo a la derecha de la imagen. Un giro de 180 grados no
cambia la lateralidad, así que los IDs de las etiquetas no se tocan.

La lista sale de leer la cabecera de los 100 archivos de imagen (las etiquetas traen la misma
orientación que su imagen); ``tests/test_orientacion.py`` la comprueba cuando data/raw existe.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

LPI_CASES = frozenset("""
002 003 004 006 009 014 023 025 028 033 035 036 041 044 045 050 056 057 058 060 062 063 067 071
072 078 079 081 082 085 086 087 088 099
""".split())


def is_rotated(case_id: str | int) -> bool:
    return str(case_id).zfill(3) in LPI_CASES


def to_canonical(volume: np.ndarray, case_id: str | int) -> np.ndarray:
    """Gira 180 grados en el plano (últimos dos ejes) los casos guardados en LPI."""
    if not is_rotated(case_id):
        return volume
    return np.ascontiguousarray(volume[..., ::-1, ::-1])


from_canonical = to_canonical   # el giro de 180 grados es su propia inversa


def header_is_rotated(path: str | Path) -> bool | None:
    """Lee la cabecera de texto de un .mha y dice si está guardado girado 180 grados (LPI).

    Sirve para volúmenes que no son de los 100 de PENGWIN, donde ``LPI_CASES`` no aplica.
    Devuelve None si la cabecera no trae ``TransformMatrix``.
    """
    with open(path, "rb") as stream:
        for line in stream.read(4096).split(b"\n"):
            key, _, value = line.decode("ascii", errors="replace").partition("=")
            if key.strip() == "TransformMatrix":
                matrix = [float(v) for v in value.split()]
                return matrix[0] < 0 and matrix[4] < 0
            if key.strip() == "ElementDataFile":
                break
    return None


def flip_inplane(volume: np.ndarray) -> np.ndarray:
    """El giro de ``to_canonical`` sin consultar la lista: para cuando la orientación sale de la cabecera."""
    return np.ascontiguousarray(volume[..., ::-1, ::-1])


def estimate_angle(images_u8: np.ndarray, mm_y: float = 1.0, mm_x: float = 1.0,
                   bone_threshold: int = 70, step: int = 3) -> tuple[float, dict]:
    """Ángulo del paciente en el plano axial, con umbrales y geometría (sin red ni etiquetas).

    Sirve para detectar pacientes acostados de lado o volúmenes girados antes de entrar al modelo.
    0 = orientación habitual (sacro abajo); +-180 = volumen guardado al revés; valores intermedios =
    paciente girado.

    1. Hueso = píxeles por encima de ``bone_threshold`` en la imagen ya ventaneada (unos 270 HU).
    2. La pelvis es mucho más ancha que profunda: la dirección principal del hueso (PCA sobre
       todo el volumen) es el eje izquierda-derecha.
    3. El sacro y la columna van en la línea media y detrás de los coxales y fémures: el hueso
       central queda desplazado hacia atrás respecto al hueso lateral. Eso fija el sentido.

    ``mm_y`` y ``mm_x`` son los milímetros por píxel del caché en cada eje (el caché estira cada
    eje distinto); sin ellos el ángulo sale deformado en los casos no cuadrados.
    ``detalle["separacion_mm"]`` pequeño (< 5 mm) indica que el sentido no es fiable.
    """
    bone = np.argwhere(images_u8[::step] > bone_threshold)
    if len(bone) < 500:
        return float("nan"), {}
    points = np.stack([bone[:, 2] * mm_x, bone[:, 1] * mm_y], axis=1).astype(np.float64)
    points -= points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    left_right = vectors[:, 1]
    back = np.array([-left_right[1], left_right[0]])
    u, t = points @ left_right, points @ back
    spread = u.std()
    central, lateral = np.abs(u) < 0.35 * spread, np.abs(u) > 0.9 * spread
    if not central.any() or not lateral.any():
        return float("nan"), {}
    separation = float(t[central].mean() - t[lateral].mean())
    if separation < 0:
        back = -back
    return math.degrees(math.atan2(back[0], back[1])), {
        "alargamiento": float(np.sqrt(values[1] / max(values[0], 1e-6))),
        "separacion_mm": abs(separation)}
