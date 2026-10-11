"""Construye un cache compacto de cortes CT para el primer entrenamiento Semana 9.

El cache evita releer volúmenes 3D completos en cada época. Las imágenes se
ventanean y las máscaras se redimensionan con la misma lógica de pengwin_io.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pengwin_io import GAMMA, load_case, resize_label_slice, resize_slice, window_hu


def choose_slices(label: np.ndarray, count: int, extent: tuple[float, float] = (0.15, 0.85),
                  negatives: int = 0) -> np.ndarray:
    """Cortes distribuidos en la extensión axial con anatomía etiquetada.

    ``extent`` fija el rango (fracción de la extensión ósea) del que se toman los
    cortes con hueso; (0.0, 1.0) cubre también los extremos, donde faltan
    regiones. ``negatives`` agrega cortes SIN hueso repartidos por el volumen,
    para que el clasificador de presencia vea ejemplos de ausencia.
    """
    has_bone = (label != 0).any(axis=(1, 2))
    valid = np.flatnonzero(has_bone)
    if len(valid) == 0:
        return np.array([], dtype=int)
    positions = np.linspace(extent[0], extent[1], min(count, len(valid)))
    chosen = valid[np.unique(np.round(positions * (len(valid) - 1)).astype(int))]
    empty = np.flatnonzero(~has_bone)
    if negatives > 0 and len(empty) > 0:
        picks = np.unique(np.round(np.linspace(0, len(empty) - 1, min(negatives, len(empty)))).astype(int))
        chosen = np.concatenate([chosen, empty[picks]])
    return np.sort(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--output", default="data/cache/week9_detection_slices.npz")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--slices-per-case", type=int, default=4)
    parser.add_argument("--max-train-cases", type=int, default=24)
    parser.add_argument("--max-val-cases", type=int, default=15)
    parser.add_argument("--extent", type=float, nargs=2, default=(0.15, 0.85),
                        help="Rango de la extensión ósea a muestrear, p. ej. 0.0 1.0.")
    parser.add_argument("--negatives-per-case", type=int, default=0,
                        help="Cortes sin hueso por caso (ejemplos de ausencia para clasificación).")
    parser.add_argument("--context", type=int, default=0,
                        help="Entrada 2.5D: K cortes vecinos a cada lado como canales (0 = 2D).")
    parser.add_argument("--canonical", action="store_true",
                        help="Aplica la misma orientación canónica usada por Semana 10.")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help=f"correccion gamma de la entrada (por defecto {GAMMA}, la de produccion); "
                             "1.0 = solo ventaneo, para comparar preprocesamientos")
    parser.add_argument("--max-test-cases", type=int, default=0,
                        help="Incluye test solo para evaluación final; 0 conserva el cache de entrenamiento.")
    args = parser.parse_args()

    with open(args.splits) as stream:
        splits = json.load(stream)
    selected = {
        "train": [str(cid).zfill(3) for cid in splits["train"][:args.max_train_cases]],
        "val": [str(cid).zfill(3) for cid in splits["val"][:args.max_val_cases]],
        "test": [str(cid).zfill(3) for cid in splits["test"][:args.max_test_cases]],
    }
    images, labels, case_ids, z_indices, split_names, spacing_yx = [], [], [], [], [], []
    for split, ids in selected.items():
        for case_id in tqdm(ids, desc=f"Preparando {split}"):
            case = load_case(case_id, load_label=True)
            if case.label is None:
                raise RuntimeError(f"El caso {case_id} no tiene máscara")
            image, label = case.image, case.label
            if args.canonical:
                from orientacion import canonicalize_arrays
                image, label, spacing_xyz, _ = canonicalize_arrays(
                    image, label, case.direction_xyz, case.spacing_xyz)
            else:
                spacing_xyz = case.spacing_xyz
            mm_y = float(spacing_xyz[1]) * image.shape[1] / args.size
            mm_x = float(spacing_xyz[0]) * image.shape[2] / args.size
            for z in choose_slices(label, args.slices_per_case, tuple(args.extent),
                                   args.negatives_per_case):
                depth = image.shape[0]
                stack = [resize_slice(window_hu(image[min(max(z + d, 0), depth - 1)], gamma=args.gamma),
                                      args.size).astype(np.float16)
                         for d in range(-args.context, args.context + 1)]
                images.append(np.stack(stack))  # (2K+1, S, S); K=0 → (1, S, S)
                labels.append(resize_label_slice(label[z], args.size).astype(np.uint8))
                case_ids.append(case_id)
                z_indices.append(z)
                split_names.append(split)
                spacing_yx.append((mm_y, mm_x))
    if not images:
        raise RuntimeError("No se encontraron cortes etiquetados")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=np.stack(images), labels=np.stack(labels),
        case_ids=np.array(case_ids), z_indices=np.array(z_indices),
        split_names=np.array(split_names), size=np.array(args.size),
        spacing_yx=np.asarray(spacing_yx, dtype=np.float32),
        context=np.array(args.context), canonical=np.array(args.canonical), gamma=np.array(args.gamma),
    )
    print(f"Cache: {out} | gamma={args.gamma} | context={args.context} | canonical={args.canonical} "
          f"| cortes={len(images)} | train={sum(s == 'train' for s in split_names)} "
          f"| val={sum(s == 'val' for s in split_names)} | test={sum(s == 'test' for s in split_names)}")


if __name__ == "__main__":
    main()
