"""Caché de test con cortes de TODO el volumen (con y sin hueso).

Sirve solo para estimar el AUC de clasificación de presencia en test: el caché
estándar toma cortes del 15–85 % de la extensión ósea y casi no tiene negativos
(0 y 1 de 180 para los coxales), así que el AUC no es calculable ahí. Las
métricas de segmentación de este caché no se reportan (incluye cortes vacíos).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pengwin_io import load_case, resize_label_slice, resize_slice, window_hu

SIZE, SLICES_PER_CASE = 256, 16
OUT = Path("data/cache/week10_test_fullrange.npz")

splits = json.load(open("data/splits.json"))
images, labels, case_ids, z_indices, split_names = [], [], [], [], []
for cid in splits["test"]:
    cid = str(cid).zfill(3)
    case = load_case(cid, load_label=True)
    for z in np.linspace(0, case.image.shape[0] - 1, SLICES_PER_CASE).round().astype(int):
        images.append(resize_slice(window_hu(case.image[z]), SIZE).astype(np.float16))
        labels.append(resize_label_slice(case.label[z], SIZE).astype(np.uint8))
        case_ids.append(cid); z_indices.append(int(z)); split_names.append("test")
    print(f"caso {cid}: {case.image.shape[0]} cortes en total, {SLICES_PER_CASE} muestreados")

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, images=np.stack(images)[:, None], labels=np.stack(labels),
                    case_ids=np.array(case_ids), z_indices=np.array(z_indices),
                    split_names=np.array(split_names), size=np.array(SIZE))
print(f"Cache: {OUT} | cortes={len(images)}")
