"""
make_splits.py
--------------
Genera splits fijos train/val/test a partir de data/manifest.csv.
- Split por case_id (nunca por corte individual).
- Estratificado por número de fragmentos (bucket bajo/medio/alto), para que
  las tres particiones tengan una distribución de dificultad comparable.
- Semilla fija para reproducibilidad.

Uso:
    python make_splits.py
"""

import json
import pandas as pd
import numpy as np

MANIFEST_PATH = "data/manifest.csv"
SPLITS_PATH = "data/splits.json"

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (el resto)

rng = np.random.default_rng(SEED)


def bucket_fragmentos(n):
    """Bucket simple de dificultad según número de fragmentos."""
    if n <= 3:
        return "bajo"
    elif n <= 6:
        return "medio"
    else:
        return "alto"


def split_group(ids, train_frac, val_frac, rng):
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
    return train, val, test


def main():
    df = pd.read_csv(MANIFEST_PATH)
    validos = df[df["valido"]].copy()

    validos["bucket"] = validos["n_fragmentos_unicos"].apply(bucket_fragmentos)

    train_ids, val_ids, test_ids = [], [], []

    for bucket, group in validos.groupby("bucket"):
        t, v, te = split_group(group["case_id"].tolist(), TRAIN_FRAC, VAL_FRAC, rng)
        train_ids += t
        val_ids += v
        test_ids += te
        print(f"Bucket '{bucket}': {len(group)} casos -> train={len(t)}, val={len(v)}, test={len(te)}")

    splits = {
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nTotal: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Splits guardados en: {SPLITS_PATH}")


if __name__ == "__main__":
    main()
