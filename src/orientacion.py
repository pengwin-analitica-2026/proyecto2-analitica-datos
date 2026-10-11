"""Orientación física y enderezado anatómico de los volúmenes PENGWIN.

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

Para archivos nuevos no se usa el nombre del caso: ``analyse_direction`` valida los tres ejes
de ``TransformMatrix`` con productos punto y cruz y ``canonicalize_arrays`` lleva la imagen a
dirección identidad mediante permutaciones/inversiones físicas de SimpleITK. Es la adaptación a
MetaImage del primer paso de Ferro Sánchez et al. (MethodsX 2024, doi:10.1016/j.mex.2024.103073).

``normalize_patient_rotation`` cubre un problema diferente: una cabecera correcta no dice si el
paciente quedó girado dentro del escáner. Busca el eje de simetría izquierda-derecha de la pelvis
y solo corrige cuando la mejora es clara y no recorta el campo de visión. La anatomía craneal del
artículo no se reutiliza; esta parte es deliberadamente específica de pelvis.
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


def read_transform_matrix(path: str | Path) -> np.ndarray | None:
    """Lee ``TransformMatrix`` de una cabecera MHA sin cargar sus cientos de MB."""
    with open(path, "rb") as stream:
        for line in stream.read(4096).split(b"\n"):
            key, _, value = line.decode("ascii", errors="replace").partition("=")
            if key.strip() == "TransformMatrix":
                matrix = [float(v) for v in value.split()]
                if len(matrix) != 9:
                    raise ValueError(f"TransformMatrix debe tener 9 valores: {path}")
                return np.asarray(matrix, dtype=np.float64).reshape(3, 3)
            if key.strip() == "ElementDataFile":
                break
    return None


def analyse_direction(direction: np.ndarray, tolerance: float = 1e-4) -> dict:
    """Valida una matriz de dirección 3D mediante punto y cruz.

    En ITK las columnas son los ejes físicos x, y, z de la imagen. Los productos punto deben ser
    cero entre ejes distintos y el producto cruz de x por y debe coincidir con z. Una matriz
    ortonormal puede reorientarse sin deformar el volumen; una matriz inválida se rechaza.
    """
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    axes = [matrix[:, index] for index in range(3)]
    norms = np.asarray([np.linalg.norm(axis) for axis in axes])
    unit = [axis / max(norm, 1e-12) for axis, norm in zip(axes, norms)]
    dots = np.asarray([[float(np.dot(a, b)) for b in unit] for a in unit])
    cross = np.cross(unit[0], unit[1])
    handedness = float(np.dot(cross, unit[2]))
    determinant = float(np.linalg.det(matrix))
    orthonormal = bool(np.allclose(norms, 1.0, atol=tolerance) and
                       np.allclose(dots, np.eye(3), atol=tolerance) and
                       abs(abs(handedness) - 1.0) <= tolerance)
    axis_aligned = bool(orthonormal and
                        np.allclose(np.max(np.abs(matrix), axis=0), 1.0, atol=tolerance) and
                        np.all((np.abs(matrix) > 1 - tolerance).sum(axis=0) == 1))
    return {"matrix": matrix, "norms": norms, "dots": dots, "cross_xy": cross,
            "handedness": handedness, "determinant": determinant,
            "orthonormal": orthonormal, "axis_aligned": axis_aligned,
            "identity": bool(np.allclose(matrix, np.eye(3), atol=tolerance))}


def header_is_rotated(path: str | Path) -> bool | None:
    """Compatibilidad: detecta el giro LPI de 180° presente en PENGWIN.

    Para cualquier otra orientación devuelve ``None``: un booleano no puede describir una
    permutación u orientación oblicua y no debe provocar una corrección silenciosa incorrecta.
    """
    matrix = read_transform_matrix(path)
    if matrix is None:
        return None
    if np.allclose(matrix, np.eye(3), atol=1e-4):
        return False
    lpi = np.diag([-1.0, -1.0, 1.0])
    return True if np.allclose(matrix, lpi, atol=1e-4) else None


def canonicalize_arrays(volume: np.ndarray, label: np.ndarray | None,
                        direction: np.ndarray | tuple[float, ...],
                        spacing_xyz: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray | None,
                                                                         tuple[float, float, float], dict]:
    """Reorienta arrays ``[z,y,x]`` a dirección identidad (LPS físico de ITK).

    SimpleITK aplica únicamente permutaciones e inversiones de ejes: no interpola intensidades
    ni IDs. Se exige una dirección ortonormal; una matriz oblicua válida conserva su dirección y
    se rechaza aquí porque llevarla a identidad sí exigiría remuestreo 3D.
    """
    info = analyse_direction(np.asarray(direction).reshape(3, 3))
    if not info["orthonormal"]:
        raise ValueError("TransformMatrix no es ortonormal; no se puede normalizar con seguridad")
    if not info["axis_aligned"]:
        raise ValueError("TransformMatrix oblicua: requiere remuestreo 3D explícito")
    if info["identity"]:
        return volume, label, tuple(float(v) for v in spacing_xyz), {
            "applied": False, "method": "metadata_dot_cross", "determinant": info["determinant"]}

    import SimpleITK as sitk

    def orient(array: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(tuple(float(v) for v in spacing_xyz))
        image.SetDirection(tuple(float(v) for v in info["matrix"].reshape(-1)))
        oriented = sitk.DICOMOrient(image, "LPS")
        return np.ascontiguousarray(sitk.GetArrayFromImage(oriented)), tuple(oriented.GetSpacing())

    result, spacing = orient(volume)
    result_label = orient(label)[0] if label is not None else None
    return result, result_label, spacing, {
        "applied": True, "method": "metadata_dot_cross", "determinant": info["determinant"],
        "source_direction": info["matrix"].reshape(-1).tolist(), "target_direction": np.eye(3).reshape(-1).tolist()}


def flip_inplane(volume: np.ndarray) -> np.ndarray:
    """El giro de ``to_canonical`` sin consultar la lista: para cuando la orientación sale de la cabecera."""
    return np.ascontiguousarray(volume[..., ::-1, ::-1])


def rotate_inplane_physical(volume: np.ndarray, angle_degrees: float, mm_y: float, mm_x: float,
                            order: int, cval: float = 0.0) -> np.ndarray:
    """Gira todos los cortes en milímetros y conserva la forma ``[z,y,x]``.

    Un ángulo positivo mueve el contenido en sentido antihorario en la imagen. La matriz incluye
    ``mm_y/mm_x``; por eso no deforma anatomía cuando los píxeles no son cuadrados.
    """
    from scipy import ndimage

    if volume.ndim != 3:
        raise ValueError("Se espera un volumen [z,y,x]")
    if mm_y <= 0 or mm_x <= 0:
        raise ValueError("Los espaciamientos deben ser positivos")
    angle = math.radians(float(angle_degrees))
    cos, sin = math.cos(angle), math.sin(angle)
    inverse_physical = np.asarray([[cos, sin], [-sin, cos]], dtype=np.float64)
    scale = np.diag([float(mm_y), float(mm_x)])
    matrix = np.linalg.inv(scale) @ inverse_physical @ scale
    center = (np.asarray(volume.shape[-2:], dtype=np.float64) - 1.0) / 2.0
    offset = center - matrix @ center
    result = np.empty_like(volume)
    for z in range(len(volume)):
        result[z] = ndimage.affine_transform(
            volume[z], matrix, offset=offset, output_shape=volume.shape[-2:], order=order,
            mode="constant", cval=float(cval), prefilter=order > 1)
    return result


def _symmetry_score(projection: np.ndarray) -> float:
    """Similitud izquierda-derecha permitiendo que la pelvis no esté centrada en el FOV."""
    from scipy import ndimage

    weights = np.asarray(projection, dtype=np.float32)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    x_center = float((weights.sum(axis=0) * np.arange(weights.shape[1])).sum() / total)
    shift = (weights.shape[1] - 1) / 2.0 - x_center
    centered = ndimage.shift(weights, (0.0, shift), order=1, mode="constant", cval=0.0, prefilter=False)
    mirror = centered[:, ::-1]
    union = np.maximum(centered, mirror).sum()
    return float(1.0 - np.abs(centered - mirror).sum() / max(float(union), 1e-6))


def estimate_pelvis_correction(images_u8: np.ndarray, mm_y: float, mm_x: float,
                               bone_threshold: int = 70, max_angle: float = 70.0,
                               step_degrees: float = 1.0) -> dict:
    """Busca el giro que maximiza la simetría de la silueta ósea axial de la pelvis."""
    from scipy import ndimage

    sampled = images_u8[::max(1, len(images_u8) // 96)]
    bone = sampled > bone_threshold
    if int(bone.sum()) < 500:
        return {"accepted": False, "reason": "insufficient_bone", "angle_degrees": 0.0}
    # Frecuencia, no máximo: reduce la influencia de metal y fragmentos aislados.
    projection = bone.mean(axis=0).astype(np.float32)
    projection = ndimage.gaussian_filter(projection, sigma=1.2)
    # La búsqueda evalúa hasta 141 ángulos; 192 px conservan la silueta y reducen
    # aproximadamente 4× el coste al cargar los 85 casos de train/validación.
    if max(projection.shape) > 192:
        old_h, old_w = projection.shape
        factor = 192.0 / max(old_h, old_w)
        new_h, new_w = max(32, round(old_h * factor)), max(32, round(old_w * factor))
        projection = ndimage.zoom(projection, (new_h / old_h, new_w / old_w), order=1,
                                  mode="nearest", prefilter=False)
        mm_y *= old_h / new_h
        mm_x *= old_w / new_w
    reference_mass = float(projection.sum())
    angles = np.arange(-max_angle, max_angle + step_degrees * 0.5, step_degrees)
    scores, retained = [], []
    stack = projection[None]
    for angle in angles:
        turned = rotate_inplane_physical(stack, float(angle), mm_y, mm_x, order=1)[0]
        scores.append(_symmetry_score(turned))
        retained.append(float(turned.sum()) / max(reference_mass, 1e-6))
    scores = np.asarray(scores)
    retained = np.asarray(retained)
    best_index = int(np.argmax(scores))
    zero_index = int(np.argmin(np.abs(angles)))
    angle = float(angles[best_index])
    improvement = float(scores[best_index] - scores[zero_index])
    result = {"angle_degrees": angle, "score_before": float(scores[zero_index]),
              "score_after": float(scores[best_index]), "improvement": improvement,
              "retained_fraction": float(retained[best_index]), "accepted": False,
              "method": "pelvis_symmetry_mm"}
    if abs(angle) <= 5.0:
        result["reason"] = "angle_below_5deg"
    elif abs(angle) >= max_angle - step_degrees * 0.5:
        result["reason"] = "search_boundary"
    elif improvement < 0.02:
        result["reason"] = "small_symmetry_improvement"
    elif scores[best_index] < 0.35:
        result["reason"] = "low_absolute_symmetry"
    elif retained[best_index] < 0.97:
        result["reason"] = "field_of_view_crop_risk"
    else:
        result["accepted"] = True
        result["reason"] = "clear_symmetry_improvement"
    return result


def normalize_patient_rotation(images_u8: np.ndarray, labels: np.ndarray | None,
                               mm_y: float, mm_x: float, **kwargs) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Endereza la pelvis solo cuando ``estimate_pelvis_correction`` acepta la corrección."""
    info = estimate_pelvis_correction(images_u8, mm_y, mm_x, **kwargs)
    if not info["accepted"]:
        return images_u8, labels, info
    images = rotate_inplane_physical(images_u8, info["angle_degrees"], mm_y, mm_x, order=1)
    out_labels = None if labels is None else rotate_inplane_physical(
        labels, info["angle_degrees"], mm_y, mm_x, order=0)
    if images_u8.dtype == np.uint8:
        images = np.clip(np.rint(images), 0, 255).astype(np.uint8)
    if labels is not None:
        out_labels = out_labels.astype(labels.dtype, copy=False)
    return images, out_labels, info


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
