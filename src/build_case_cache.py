"""Caché por caso con el volumen completo, para entrenar y evaluar la Semana 10.

A diferencia de ``prepare_week9_detection_data.py`` (que muestrea unos pocos
cortes del centro del hueso y los junta en un solo archivo), aquí se guarda
**cada corte de cada caso**, incluidos los extremos y los cortes sin hueso, en
un archivo por caso. Así:

- el entrenamiento puede elegir qué cortes usar sin regenerar el caché;
- la evaluación recorre el volumen completo, que es lo que verá la demo;
- las distancias en mm se miden contra la etiqueta en resolución nativa.

El preprocesamiento es el mismo de ``pengwin_io`` (ventana ósea + gamma, luego
resize bilineal; vecino más cercano para etiquetas). La imagen se guarda en
uint8 (0-255) para ahorrar espacio: el error de cuantización es de 1/255 como
máximo y se deshace dividiendo por 255 al cargar.

Dos modos de llevar el corte a un cuadrado de lado S:

- ``stretch``: resize directo, como en las semanas 9 y 10. Deforma los casos
  que no son cuadrados (hay volúmenes de 154x322).
- ``pad``: rellena con fondo hasta un cuadrado y luego redimensiona, así la
  anatomía conserva su proporción en todos los casos.

Salida en ``--out-dir``:

    cache_<size>/<caso>.npz       modo stretch: images (Z,S,S) uint8, labels (Z,S,S) uint8
    cache_<size>_pad/<caso>.npz   modo pad, mismo contenido más ``pad_yx``
    labels_native/<caso>.npz  label (Z,Y,X) uint8 en resolución original
    index.csv                 una fila por caso: split, forma, spacing, cortes

Uso:

    python src/build_case_cache.py --out-dir data/cache/s10 --sizes 256 384
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pengwin_io
from pengwin_io import GAMMA, load_case, resize_label_slice, resize_slice, window_hu

INDEX_FIELDS = ["case_id", "split", "shape_z", "shape_y", "shape_x",
                "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
                "n_slices", "n_bone_slices", "first_bone_z", "last_bone_z", "n_fragments"]


def to_uint8(image01: np.ndarray) -> np.ndarray:
    """[0, 1] float -> uint8 con redondeo (no truncado)."""
    return np.clip(np.rint(image01 * 255.0), 0, 255).astype(np.uint8)


def from_uint8(image_u8: np.ndarray) -> np.ndarray:
    """Inversa de ``to_uint8``: uint8 -> float32 en [0, 1]."""
    return image_u8.astype(np.float32) / 255.0


def pad_to_square(slice_2d: np.ndarray, fill: float = 0) -> tuple[np.ndarray, tuple[int, int]]:
    """Centra el corte en un cuadrado relleno con ``fill``. Devuelve (corte, (pad_y, pad_x))."""
    height, width = slice_2d.shape
    side = max(height, width)
    pad_y, pad_x = (side - height) // 2, (side - width) // 2
    out = np.full((side, side), fill, dtype=slice_2d.dtype)
    out[pad_y:pad_y + height, pad_x:pad_x + width] = slice_2d
    return out, (pad_y, pad_x)


def cache_dir_name(size: int, mode: str) -> str:
    return f"cache_{size}" if mode == "stretch" else f"cache_{size}_{mode}"


def split_of(case_id: str, splits: dict) -> str:
    number = int(case_id)
    for name in ("train", "val", "test"):
        if number in splits.get(name, []):
            return name
    return "sin_split"


def build_case(case_id: str, out_dir: str, sizes: tuple[int, ...], gamma: float,
               split: str, raw_root: str | None = None, overwrite: bool = False,
               modes: tuple[str, ...] = ("stretch",)) -> dict:
    """Procesa un caso y devuelve su fila del índice."""
    if raw_root is not None:
        set_raw_root(raw_root)
    out = Path(out_dir)
    case_id = str(case_id).zfill(3)
    case = load_case(case_id, load_label=True)
    if case.label is None:
        raise RuntimeError(f"El caso {case_id} no tiene máscara")
    if case.label.shape != case.image.shape:
        raise RuntimeError(f"Caso {case_id}: imagen {case.image.shape} y etiqueta {case.label.shape} no coinciden")
    if int(case.label.max()) > 255:
        raise RuntimeError(f"Caso {case_id}: etiqueta fuera de uint8 ({case.label.max()})")
    label = case.label.astype(np.uint8)
    depth = case.image.shape[0]

    for mode in modes:
        if mode not in ("stretch", "pad"):
            raise ValueError(f"modo desconocido: {mode}")
        for size in sizes:
            target = out / cache_dir_name(size, mode) / f"{case_id}.npz"
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            images = np.empty((depth, size, size), dtype=np.uint8)
            labels = np.empty((depth, size, size), dtype=np.uint8)
            pad_yx = (0, 0)
            for z in range(depth):
                windowed = window_hu(case.image[z], gamma=gamma)
                mask = label[z]
                if mode == "pad":
                    # El relleno vale 0 en la ventana, igual que el aire fuera del paciente.
                    windowed, pad_yx = pad_to_square(windowed, 0.0)
                    mask, _ = pad_to_square(mask, 0)
                images[z] = to_uint8(resize_slice(windowed, size))
                labels[z] = resize_label_slice(mask, size)
            np.savez_compressed(target, images=images, labels=labels,
                                spacing_xyz=np.asarray(case.spacing_xyz, dtype=np.float64),
                                direction_xyz=np.asarray(case.direction_xyz, dtype=np.float64),
                                orig_shape=np.asarray(case.image.shape), size=np.array(size),
                                gamma=np.array(gamma), case_id=np.array(case_id),
                                mode=np.array(mode), pad_yx=np.asarray(pad_yx))

    native = out / "labels_native" / f"{case_id}.npz"
    if overwrite or not native.exists():
        native.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(native, label=label,
                            spacing_xyz=np.asarray(case.spacing_xyz, dtype=np.float64),
                            direction_xyz=np.asarray(case.direction_xyz, dtype=np.float64),
                            case_id=np.array(case_id))

    has_bone = (label != 0).any(axis=(1, 2))
    bone_z = np.flatnonzero(has_bone)
    sx, sy, sz = (float(v) for v in case.spacing_xyz)
    return {
        "case_id": case_id, "split": split,
        "shape_z": depth, "shape_y": case.image.shape[1], "shape_x": case.image.shape[2],
        "spacing_x_mm": sx, "spacing_y_mm": sy, "spacing_z_mm": sz,
        "n_slices": depth, "n_bone_slices": int(has_bone.sum()),
        "first_bone_z": int(bone_z[0]) if len(bone_z) else -1,
        "last_bone_z": int(bone_z[-1]) if len(bone_z) else -1,
        "n_fragments": int(len(np.unique(label)) - 1),
    }


def set_raw_root(raw_root: str) -> None:
    """Apunta ``pengwin_io`` a otra carpeta de datos crudos (p. ej. en Kaggle)."""
    root = Path(raw_root)
    pengwin_io.DATA_ROOT = root
    pengwin_io.IMAGES_DIRS = [root / "PENGWIN_CT_train_images_part1",
                              root / "PENGWIN_CT_train_images_part2"]
    pengwin_io.LABELS_DIR = root / "PENGWIN_CT_train_labels"


def write_index(rows: list[dict], out_dir: Path) -> Path:
    path = out_dir / "index.csv"
    existing: dict[str, dict] = {}
    if path.exists():
        with open(path, newline="") as stream:
            existing = {row["case_id"]: row for row in csv.DictReader(stream)}
    for row in rows:
        existing[row["case_id"]] = row
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for case_id in sorted(existing):
            writer.writerow(existing[case_id])
    return path


def load_case_cache(cache_dir: str | Path, case_id: str, canonical: bool = False) -> dict:
    """Lee el caché de un caso. ``images`` sigue en uint8; usar ``from_uint8``.

    ``canonical=True`` deja el caso en la orientación común (ver ``orientacion.py``).
    """
    with np.load(Path(cache_dir) / f"{str(case_id).zfill(3)}.npz") as data:
        case = {key: data[key] for key in data.files}
    if canonical:
        if "direction_xyz" in case:
            from orientacion import canonicalize_arrays
            images, labels, spacing, orientation = canonicalize_arrays(
                case["images"], case["labels"], case["direction_xyz"], tuple(case["spacing_xyz"]))
            case["images"], case["labels"] = images, labels
            case["spacing_xyz"] = np.asarray(spacing, dtype=np.float64)
            case["orientation"] = orientation
        else:
            # Compatibilidad con cachés ya construidos antes de guardar TransformMatrix.
            from orientacion import is_rotated, to_canonical
            case["images"] = to_canonical(case["images"], case_id)
            case["labels"] = to_canonical(case["labels"], case_id)
            case["orientation"] = {"applied": bool(is_rotated(case_id)), "method": "legacy_case_id"}
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-dir", default="data/cache/s10")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--modes", nargs="+", default=["stretch"], choices=["stretch", "pad"])
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Casos a procesar (por defecto, todos los de splits.json).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.splits) as stream:
        splits = json.load(stream)
    if args.cases:
        case_ids = [str(c).zfill(3) for c in args.cases]
    else:
        case_ids = [str(c).zfill(3) for name in ("train", "val", "test") for c in splits[name]]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(cid, str(out_dir), tuple(args.sizes), args.gamma, split_of(cid, splits),
             args.raw_root, args.overwrite, tuple(args.modes)) for cid in case_ids]
    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(build_case, *zip(*jobs)):
                rows.append(row)
                print(f"{row['case_id']} {row['split']:5s} cortes={row['n_slices']} con_hueso={row['n_bone_slices']}", flush=True)
    else:
        for job in jobs:
            row = build_case(*job)
            rows.append(row)
            print(f"{row['case_id']} {row['split']:5s} cortes={row['n_slices']} con_hueso={row['n_bone_slices']}", flush=True)
    index = write_index(rows, out_dir)
    print(f"Índice: {index} | casos={len(rows)} | tamaños={list(args.sizes)} | modos={list(args.modes)} | gamma={args.gamma}")


if __name__ == "__main__":
    main()
