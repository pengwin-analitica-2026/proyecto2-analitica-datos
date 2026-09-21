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
from pengwin_io import load_case, resize_label_slice, resize_slice, window_hu


def choose_slices(label: np.ndarray, count: int) -> np.ndarray:
    """Cortes distribuidos en la extensión axial con anatomía etiquetada."""
    valid = np.flatnonzero((label != 0).any(axis=(1, 2)))
    if len(valid) == 0:
        return np.array([], dtype=int)
    positions = np.linspace(0.15, 0.85, min(count, len(valid)))
    return valid[np.unique(np.round(positions * (len(valid) - 1)).astype(int))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--output", default="data/cache/week9_detection_slices.npz")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--slices-per-case", type=int, default=4)
    parser.add_argument("--max-train-cases", type=int, default=24)
    parser.add_argument("--max-val-cases", type=int, default=15)
    args = parser.parse_args()

    with open(args.splits) as stream:
        splits = json.load(stream)
    selected = {
        "train": [str(cid).zfill(3) for cid in splits["train"][:args.max_train_cases]],
        "val": [str(cid).zfill(3) for cid in splits["val"][:args.max_val_cases]],
    }
    images, labels, case_ids, z_indices, split_names = [], [], [], [], []
    for split, ids in selected.items():
        for case_id in tqdm(ids, desc=f"Preparando {split}"):
            case = load_case(case_id, load_label=True)
            if case.label is None:
                raise RuntimeError(f"El caso {case_id} no tiene máscara")
            for z in choose_slices(case.label, args.slices_per_case):
                images.append(resize_slice(window_hu(case.image[z]), args.size).astype(np.float16))
                labels.append(resize_label_slice(case.label[z], args.size).astype(np.uint8))
                case_ids.append(case_id)
                z_indices.append(z)
                split_names.append(split)
    if not images:
        raise RuntimeError("No se encontraron cortes etiquetados")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=np.stack(images)[:, None], labels=np.stack(labels),
        case_ids=np.array(case_ids), z_indices=np.array(z_indices),
        split_names=np.array(split_names), size=np.array(args.size),
    )
    print(f"Cache: {out} | cortes={len(images)} | train={sum(s == 'train' for s in split_names)} "
          f"| val={sum(s == 'val' for s in split_names)}")


if __name__ == "__main__":
    main()

