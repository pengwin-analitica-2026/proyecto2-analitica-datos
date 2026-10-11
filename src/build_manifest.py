"""
build_manifest.py
------------------
Construye el manifest del dataset PENGWIN (Task 1 - CT) a partir de archivos .mha.

Qué hace:
1. Recorre las carpetas de imágenes (part1 + part2) y de labels.
2. Para cada caso, carga imagen y máscara con SimpleITK.
3. Extrae metadatos completos: shape, spacing (mm/vóxel), tipo de dato,
   rango de intensidad (HU), número de fragmentos únicos, y si el par
   imagen-label está completo y consistente.
4. Guarda todo en data/manifest.csv.

Requisitos:
    pip install SimpleITK pandas numpy tqdm

Uso:
    python build_manifest.py
"""

from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURACIÓN — ajusta esta ruta si moviste los datos a otro lugar
# ---------------------------------------------------------------------------
DATA_ROOT = Path("data/raw")

IMAGES_DIRS = [
    DATA_ROOT / "PENGWIN_CT_train_images_part1",
    DATA_ROOT / "PENGWIN_CT_train_images_part2",
]
LABELS_DIR = DATA_ROOT / "PENGWIN_CT_train_labels"

# Carpeta de salida del proyecto (ajusta a la ruta real de tu repo si corres
# este script desde otro lugar)
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"


# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------
def collect_case_ids(images_dirs, labels_dir):
    """Recorre las carpetas de imágenes y labels y arma un dict case_id -> rutas."""
    cases = {}

    for img_dir in images_dirs:
        if not img_dir.exists():
            print(f"[AVISO] No existe la carpeta: {img_dir}")
            continue
        for f in img_dir.glob("*.mha"):
            case_id = f.stem  # ej. "001"
            cases.setdefault(case_id, {})["img_path"] = f

    if not labels_dir.exists():
        print(f"[AVISO] No existe la carpeta de labels: {labels_dir}")
    else:
        for f in labels_dir.glob("*.mha"):
            case_id = f.stem
            cases.setdefault(case_id, {})["label_path"] = f

    return cases


def inspect_case(case_id, paths):
    """Carga imagen y label (si existen) y extrae metadatos completos."""
    row = {"case_id": case_id}

    img_path = paths.get("img_path")
    label_path = paths.get("label_path")

    row["img_path"] = str(img_path) if img_path else None
    row["label_path"] = str(label_path) if label_path else None
    row["tiene_imagen"] = img_path is not None
    row["tiene_label"] = label_path is not None

    # --- Imagen ---
    if img_path is not None:
        try:
            img = sitk.ReadImage(str(img_path))
            img_arr = sitk.GetArrayFromImage(img)  # (z, y, x)

            row["shape_z"], row["shape_y"], row["shape_x"] = img_arr.shape
            sx, sy, sz = img.GetSpacing()  # SimpleITK da (x, y, z)
            row["spacing_x_mm"] = sx
            row["spacing_y_mm"] = sy
            row["spacing_z_mm"] = sz
            row["img_dtype"] = str(img_arr.dtype)
            row["hu_min"] = float(img_arr.min())
            row["hu_max"] = float(img_arr.max())
            row["hu_mean"] = float(img_arr.mean())
            row["img_error"] = None
        except Exception as e:
            row["img_error"] = str(e)
    else:
        row["img_error"] = "archivo de imagen no encontrado"

    # --- Label / máscara ---
    if label_path is not None:
        try:
            label = sitk.ReadImage(str(label_path))
            label_arr = sitk.GetArrayFromImage(label)

            row["label_shape_z"], row["label_shape_y"], row["label_shape_x"] = label_arr.shape
            row["label_dtype"] = str(label_arr.dtype)

            unique_vals = np.unique(label_arr)
            unique_vals_no_bg = unique_vals[unique_vals != 0]  # excluye fondo
            row["n_fragmentos_unicos"] = int(len(unique_vals_no_bg))
            row["valores_unicos_label"] = ",".join(map(str, unique_vals_no_bg.tolist()))
            row["label_error"] = None
        except Exception as e:
            row["label_error"] = str(e)
    else:
        row["label_error"] = "archivo de label no encontrado"

    # --- Consistencia imagen-label ---
    if img_path is not None and label_path is not None and row.get("img_error") is None and row.get("label_error") is None:
        same_shape = (
            row["shape_z"] == row["label_shape_z"]
            and row["shape_y"] == row["label_shape_y"]
            and row["shape_x"] == row["label_shape_x"]
        )
        row["shapes_coinciden"] = same_shape
    else:
        row["shapes_coinciden"] = False

    # --- Flag de validez general ---
    row["valido"] = bool(
        row["tiene_imagen"]
        and row["tiene_label"]
        and row.get("img_error") is None
        and row.get("label_error") is None
        and row.get("shapes_coinciden", False)
    )

    return row


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(f"Buscando archivos en: {DATA_ROOT}")
    cases = collect_case_ids(IMAGES_DIRS, LABELS_DIR)
    print(f"Casos detectados (por nombre de archivo): {len(cases)}")

    rows = []
    for case_id in tqdm(sorted(cases.keys()), desc="Inspeccionando casos"):
        row = inspect_case(case_id, cases[case_id])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST_PATH, index=False)

    # --- Resumen en consola ---
    n_total = len(df)
    n_validos = df["valido"].sum()
    n_invalidos = n_total - n_validos

    print("\n" + "=" * 60)
    print("RESUMEN DEL MANIFEST")
    print("=" * 60)
    print(f"Total de casos encontrados : {n_total}")
    print(f"Casos válidos               : {n_validos}")
    print(f"Casos inválidos/incompletos : {n_invalidos}")

    if n_invalidos > 0:
        print("\nCasos inválidos (detalle):")
        cols_debug = ["case_id", "tiene_imagen", "tiene_label", "img_error", "label_error", "shapes_coinciden"]
        print(df[~df["valido"]][cols_debug].to_string(index=False))

    if n_validos > 0:
        validos = df[df["valido"]]
        print("\nRango de spacing (mm/vóxel) en casos válidos:")
        print(f"  spacing_x: {validos['spacing_x_mm'].min():.3f} - {validos['spacing_x_mm'].max():.3f}")
        print(f"  spacing_y: {validos['spacing_y_mm'].min():.3f} - {validos['spacing_y_mm'].max():.3f}")
        print(f"  spacing_z: {validos['spacing_z_mm'].min():.3f} - {validos['spacing_z_mm'].max():.3f}")
        print(f"\nRango HU global: {validos['hu_min'].min():.1f} a {validos['hu_max'].max():.1f}")
        print(f"\nFragmentos únicos por caso (excluyendo fondo):")
        print(f"  min: {validos['n_fragmentos_unicos'].min()}, "
              f"max: {validos['n_fragmentos_unicos'].max()}, "
              f"promedio: {validos['n_fragmentos_unicos'].mean():.2f}")

    print(f"\nManifest guardado en: {MANIFEST_PATH.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
