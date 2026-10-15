"""Post-proceso 3D y medición de distancias en mm sobre el volumen completo (Fase 6).

El modelo trabaja corte a corte: un corte aislado no sabe cuál fragmento es el principal del
hueso, ni si un fragmento pequeño que aparece en un solo corte es real. Aquí se usa el volumen:

1. por región se descartan componentes 3D diminutos de la máscara semántica (ruido);
2. el fragmento principal es el componente 3D conectado más grande de la clase "primario";
3. cada componente 3D conectado de "secundario" (y los primarios que no son el mayor) es un
   fragmento, con ID start+1, start+2... por volumen decreciente; los menores a ``min_voxels``
   se absorben en el principal, así no se pierden píxeles de la región.

Las distancias salen de ``fragment_distances`` (EDT con el spacing del header). Se agregan
ASSD y HD95 por fragmento, que el enunciado deja como opcionales.

Umbrales en mm³. ``min_voxels`` y ``min_region_voxels`` cuentan vóxeles, así que dependen del
espaciado: con cortes más gruesos un fragmento ocupa menos vóxeles y puede borrarse por "pequeño".
``umbrales_en_voxeles`` convierte umbrales en mm³ a vóxeles para cada volumen. Los valores por
defecto en mm³ equivalen a 30 y 200 vóxeles del vóxel mediano del caché de PENGWIN
(``VOXEL_REFERENCIA_MM3``).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from fragment_distances import REGIONS, edge_distance_mm

STRUCTURE_3D = np.ones((3, 3, 3), dtype=np.uint8)
REGION_STARTS = (1, 11, 21)

# Vóxel mediano del caché a 384 px de los 100 casos de PENGWIN (0.22 a 1.43 mm³ entre casos).
VOXEL_REFERENCIA_MM3 = 0.869
MIN_FRAGMENTO_MM3 = 30 * VOXEL_REFERENCIA_MM3      # 26.1 mm³, el equivalente de min_voxels=30
MIN_REGION_MM3 = 200 * VOXEL_REFERENCIA_MM3        # 173.8 mm³, el de min_region_voxels=200


def umbrales_en_voxeles(spacing_xyz: tuple[float, float, float], min_mm3: float = MIN_FRAGMENTO_MM3,
                        min_region_mm3: float = MIN_REGION_MM3) -> tuple[int, int]:
    """(min_voxels, min_region_voxels) que corresponden a umbrales en mm³ con este espaciado."""
    voxel = float(np.prod(spacing_xyz))
    return max(1, int(np.ceil(min_mm3 / voxel))), max(1, int(np.ceil(min_region_mm3 / voxel)))


def cache_spacing_xyz(row: dict, size: int) -> tuple[float, float, float]:
    """Spacing (x, y, z) en mm de un volumen del caché ``stretch`` (S×S por corte).

    El caché estira cada corte de (y, x) originales a (S, S); cada píxel cubre más o menos
    milímetros según el caso. ``row`` es una fila de ``index.csv``.
    """
    return (float(row["spacing_x_mm"]) * float(row["shape_x"]) / size,
            float(row["spacing_y_mm"]) * float(row["shape_y"]) / size,
            float(row["spacing_z_mm"]))


def postprocess_3d(semantic: np.ndarray, primary_secondary: np.ndarray, min_voxels: int = 30,
                   min_region_voxels: int = 200, fill_unlabeled: bool = True) -> np.ndarray:
    """Volumen [z,y,x] de IDs PENGWIN a partir de la semántica y del mapa de 7 clases.

    Ver el docstring del módulo. ``min_region_voxels`` descarta regiones 3D enteras que son
    ruido (por ejemplo, un coxal "detectado" en unos pocos vóxeles aislados).
    ``fill_unlabeled=False`` deja sin etiqueta los vóxeles anatómicos que no son ni principal ni
    fragmento: así no se rellena el hueco entre fragmentos y se puede medir su separación.
    """
    output = np.zeros(semantic.shape, dtype=np.uint8)
    for region, start in enumerate(REGION_STARTS):
        anatomical = semantic == region + 1
        components, n = ndi.label(anatomical, structure=STRUCTURE_3D)
        if n == 0:
            continue
        sizes = np.asarray(ndi.sum(anatomical, components, index=range(1, n + 1)))
        keep = np.flatnonzero(sizes >= min_region_voxels) + 1
        anatomical = np.isin(components, keep)
        if not anatomical.any():
            continue
        primary = (primary_secondary == 2 * region + 1) & anatomical
        secondary = (primary_secondary == 2 * region + 2) & anatomical
        labeled, n = ndi.label(primary, structure=STRUCTURE_3D)
        if n > 0:
            sizes = np.asarray(ndi.sum(primary, labeled, index=range(1, n + 1)))
            main = labeled == (int(np.argmax(sizes)) + 1)
            secondary |= primary & ~main
        else:   # sin primario predicho: el componente anatómico mayor hace de principal
            labeled, n = ndi.label(anatomical, structure=STRUCTURE_3D)
            sizes = np.asarray(ndi.sum(anatomical, labeled, index=range(1, n + 1)))
            main = labeled == (int(np.argmax(sizes)) + 1)
            secondary &= ~main
        output[main] = start
        pieces, n = ndi.label(secondary, structure=STRUCTURE_3D)
        sizes = np.asarray(ndi.sum(secondary, pieces, index=range(1, n + 1))) if n else np.array([])
        order = [int(i) + 1 for i in np.argsort(-sizes) if sizes[i] >= min_voxels][:9]
        for rank, piece in enumerate(order):
            output[pieces == piece] = start + 1 + rank
        # Vóxeles de la región sin etiqueta (secundarios chicos, huecos) → principal.
        if fill_unlabeled:
            output[anatomical & (output == 0)] = start
    return output


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1))


def surface_distances(a: np.ndarray, b: np.ndarray, spacing_xyz: tuple[float, float, float]
                      ) -> tuple[float, float]:
    """(ASSD, HD95) en mm entre las superficies de dos máscaras 3D no vacías."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or not b.any():
        raise ValueError("ambas máscaras deben contener al menos un vóxel")
    coords = np.argwhere(a | b)
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    crop = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    a, b = a[crop], b[crop]
    sampling = np.asarray(spacing_xyz[::-1], dtype=np.float64)
    surface_a, surface_b = _surface(a), _surface(b)
    to_b = ndi.distance_transform_edt(~surface_b, sampling=sampling)[surface_a]
    to_a = ndi.distance_transform_edt(~surface_a, sampling=sampling)[surface_b]
    both = np.concatenate([to_b, to_a])
    return float((to_b.mean() + to_a.mean()) / 2), float(np.percentile(both, 95))


def match_fragments(prediction: np.ndarray, truth: np.ndarray) -> list[dict]:
    """Empareja en 3D cada fragmento real con el predicho de mayor IoU de su misma región.

    Un fragmento predicho se usa una sola vez. Devuelve Dice e IoU 3D por fragmento real
    (cero si no hay pareja), con el ID predicho emparejado.
    """
    rows = []
    for low, high in ((1, 10), (11, 20), (21, 30)):
        gt_ids = [int(v) for v in np.unique(truth) if low <= v <= high]
        pred_ids = [int(v) for v in np.unique(prediction) if low <= v <= high]
        used: set[int] = set()
        for gt_id in sorted(gt_ids, key=lambda g: -int((truth == g).sum())):
            gt = truth == gt_id
            best, best_iou = None, 0.0
            for pred_id in pred_ids:
                if pred_id in used:
                    continue
                cand = prediction == pred_id
                union = np.logical_or(gt, cand).sum()
                iou = float(np.logical_and(gt, cand).sum() / max(union, 1))
                if iou > best_iou:
                    best, best_iou = pred_id, iou
            dice = 0.0
            if best is not None:
                used.add(best)
                cand = prediction == best
                dice = float(2 * np.logical_and(gt, cand).sum() / max(gt.sum() + cand.sum(), 1))
            rows.append({"gt_id": gt_id, "pred_id": best, "dice": dice, "iou": best_iou,
                         "volume_voxels": int(gt.sum()), "primary": gt_id in REGION_STARTS})
    return rows


def fragment_distance_table(prediction: np.ndarray, truth: np.ndarray,
                            spacing_xyz: tuple[float, float, float],
                            with_surface_metrics: bool = False) -> list[dict]:
    """Distancia (mm) al fragmento principal de cada fragmento secundario real y predicho.

    Cada fila es un fragmento secundario real: su distancia en el GT, la del fragmento
    predicho con el que se empareja (por IoU 3D), el error absoluto, y opcionalmente
    ASSD/HD95 entre ambos. Los no detectados quedan con ``predicted_mm`` en None.
    """
    rows = []
    matches = {m["gt_id"]: m for m in match_fragments(prediction, truth)}
    for region, (low, high, main_id) in REGIONS.items():
        gt_main = truth == main_id
        pred_main = prediction == main_id
        for gt_id in sorted(int(v) for v in np.unique(truth) if low <= v <= high and v != main_id):
            if not gt_main.any():
                continue
            gt_mask = truth == gt_id
            row = {"region": region, "gt_id": gt_id, "volume_voxels": int(gt_mask.sum()),
                   "ground_truth_mm": edge_distance_mm(gt_mask, gt_main, spacing_xyz),
                   "pred_id": None, "predicted_mm": None, "absolute_error_mm": None,
                   "dice": matches[gt_id]["dice"], "iou": matches[gt_id]["iou"]}
            pred_id = matches[gt_id]["pred_id"]
            if pred_id is not None and pred_id != main_id and pred_main.any():
                pred_mask = prediction == pred_id
                row["pred_id"] = pred_id
                row["predicted_mm"] = edge_distance_mm(pred_mask, pred_main, spacing_xyz)
                row["absolute_error_mm"] = abs(row["predicted_mm"] - row["ground_truth_mm"])
                if with_surface_metrics:
                    row["assd_mm"], row["hd95_mm"] = surface_distances(pred_mask, gt_mask, spacing_xyz)
            rows.append(row)
    return rows


def summarize_volume(prediction: np.ndarray, truth: np.ndarray) -> dict:
    """Dice/IoU 3D medio por fragmento (principales y secundarios) de un caso."""
    matches = match_fragments(prediction, truth)
    out = {"n_fragments": len(matches)}
    for label, keep in (("todos", lambda m: True), ("principal", lambda m: m["primary"]),
                        ("secundario", lambda m: not m["primary"])):
        group = [m for m in matches if keep(m)]
        if group:
            out[f"dice_{label}"] = float(np.mean([m["dice"] for m in group]))
            out[f"iou_{label}"] = float(np.mean([m["iou"] for m in group]))
            out[f"n_{label}"] = len(group)
            out[f"perdidos_{label}"] = float(np.mean([m["dice"] == 0.0 for m in group]))
    return out
