"""Distancias físicas entre fragmentos PENGWIN.

Calcula la separación mínima borde a borde entre cada fragmento conminuto y
el fragmento principal de su región usando EDT y spacing (x,y,z) del header.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi


REGIONS = {
    "sacro": (1, 10, 1),
    "coxal_izquierdo": (11, 20, 11),
    "coxal_derecho": (21, 30, 21),
}


@dataclass(frozen=True)
class FragmentDistance:
    region: str
    fragment_label: int
    main_label: int
    distance_mm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def edge_distance_mm(fragment_mask: np.ndarray, main_mask: np.ndarray,
                     spacing_xyz: tuple[float, float, float]) -> float:
    """Distancia mínima entre las superficies de vóxeles binarios.

    EDT entrega el vóxel principal más cercano a cada punto. Para convertir
    distancia entre centros a borde-a-borde se descuenta un ancho de vóxel en
    cada eje con desplazamiento. Dos vóxeles adyacentes tienen separación 0.
    """
    fragment_mask = np.asarray(fragment_mask, dtype=bool)
    main_mask = np.asarray(main_mask, dtype=bool)
    if fragment_mask.shape != main_mask.shape:
        raise ValueError("fragment_mask y main_mask deben tener la misma forma")
    if not fragment_mask.any() or not main_mask.any():
        raise ValueError("ambas máscaras deben contener al menos un vóxel")
    if len(spacing_xyz) != 3 or any(float(value) <= 0 for value in spacing_xyz):
        raise ValueError("spacing_xyz debe contener tres valores positivos")
    # El volumen CT completo puede superar 100 millones de vóxeles. Recortar
    # al bounding box conjunto conserva la distancia exacta entre las dos
    # máscaras y evita que EDT reserve varios GB para índices innecesarios.
    union_coords = np.argwhere(fragment_mask | main_mask)
    lower = union_coords.min(axis=0)
    upper = union_coords.max(axis=0) + 1
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    fragment_mask = fragment_mask[crop]
    main_mask = main_mask[crop]
    spacing_zyx = np.asarray(spacing_xyz[::-1], dtype=np.float64)
    _, nearest = ndi.distance_transform_edt(
        ~main_mask, sampling=spacing_zyx, return_indices=True
    )
    fragment_coords = np.argwhere(fragment_mask)
    nearest_main = nearest[:, fragment_mask].T
    voxel_delta = np.abs(fragment_coords - nearest_main)
    # Cada máscara representa cajas de ancho spacing alrededor de sus centros.
    edge_components = np.maximum(voxel_delta - 1, 0) * spacing_zyx
    return float(np.linalg.norm(edge_components, axis=1).min())


def measure_fragment_distances(labels: np.ndarray,
                               spacing_xyz: tuple[float, float, float]
                               ) -> list[FragmentDistance]:
    """Mide todos los fragmentos no principales presentes en un volumen 3D."""
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError("labels debe ser un volumen [z,y,x]")
    records: list[FragmentDistance] = []
    for region, (low, high, main_label) in REGIONS.items():
        main_mask = labels == main_label
        present = [int(value) for value in np.unique(labels)
                   if low <= int(value) <= high and int(value) != main_label]
        if present and not main_mask.any():
            raise ValueError(f"Falta el fragmento principal {main_label} de {region}")
        for fragment_label in present:
            distance = edge_distance_mm(labels == fragment_label, main_mask, spacing_xyz)
            records.append(FragmentDistance(region, fragment_label, main_label, distance))
    return records


def compare_fragment_distances(prediction: np.ndarray, ground_truth: np.ndarray,
                               spacing_xyz: tuple[float, float, float]
                               ) -> list[dict[str, object]]:
    """Compara distancias predichas y GT para IDs de fragmento coincidentes."""
    pred = {item.fragment_label: item for item in
            measure_fragment_distances(prediction, spacing_xyz)}
    gt = {item.fragment_label: item for item in
          measure_fragment_distances(ground_truth, spacing_xyz)}
    rows = []
    for label in sorted(set(pred) | set(gt)):
        pred_mm = pred[label].distance_mm if label in pred else None
        gt_mm = gt[label].distance_mm if label in gt else None
        region = (pred.get(label) or gt[label]).region
        rows.append({
            "region": region,
            "fragment_label": label,
            "predicted_mm": pred_mm,
            "ground_truth_mm": gt_mm,
            "absolute_error_mm": None if pred_mm is None or gt_mm is None
            else abs(pred_mm - gt_mm),
        })
    return rows
