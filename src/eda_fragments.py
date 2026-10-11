"""
eda_fragments.py
-----------------
EDA de fragmentos del dataset PENGWIN, usando la convención oficial de
etiquetas (Zenodo, record 10927452):
    0        = fondo
    1-10     = fragmentos de sacro
    11-20    = fragmentos de coxal izquierdo
    21-30    = fragmentos de coxal derecho

Genera:
- data/eda_fragments_detalle.csv   (una fila por caso, con conteo por región
                                     y volumen total de hueso en mm3)
- outputs/eda/hist_fragmentos_totales.png
- outputs/eda/barras_fragmentos_por_region.png
- outputs/eda/boxplot_volumen_hueso_mm3.png
- outputs/eda/resumen_estadistico.csv

Requisitos:
    pip install SimpleITK numpy pandas matplotlib tqdm

Uso:
    python eda_fragments.py
"""

from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
MANIFEST_PATH = Path("data/manifest.csv")
LABELS_DIR = Path("data/raw") / "PENGWIN_CT_train_labels"

OUTPUT_EDA_DIR = Path("outputs/eda")
OUTPUT_EDA_DIR.mkdir(parents=True, exist_ok=True)
DETALLE_PATH = Path("data/eda_fragments_detalle.csv")


def region_de_label(label_val):
    """Clasifica un valor de label según la convención oficial de PENGWIN."""
    if 1 <= label_val <= 10:
        return "sacro"
    elif 11 <= label_val <= 20:
        return "coxal_izquierdo"
    elif 21 <= label_val <= 30:
        return "coxal_derecho"
    else:
        return "desconocido"  # no debería pasar en datos válidos


def analizar_caso(case_id, label_path, spacing_xyz):
    """Carga una máscara y devuelve conteo de fragmentos por región y volumen total en mm3."""
    label_img = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(label_img)  # (z, y, x)

    voxel_volume_mm3 = spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]

    unique_vals = np.unique(label_arr)
    unique_vals = unique_vals[unique_vals != 0]  # excluir fondo

    conteo_region = {"sacro": 0, "coxal_izquierdo": 0, "coxal_derecho": 0, "desconocido": 0}
    for val in unique_vals:
        conteo_region[region_de_label(int(val))] += 1

    n_voxeles_hueso = int((label_arr != 0).sum())
    volumen_total_mm3 = n_voxeles_hueso * voxel_volume_mm3

    return {
        "case_id": case_id,
        "n_fragmentos_total": int(len(unique_vals)),
        "n_fragmentos_sacro": conteo_region["sacro"],
        "n_fragmentos_coxal_izq": conteo_region["coxal_izquierdo"],
        "n_fragmentos_coxal_der": conteo_region["coxal_derecho"],
        "n_labels_desconocidos": conteo_region["desconocido"],
        "volumen_hueso_total_mm3": volumen_total_mm3,
    }


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    validos = manifest[manifest["valido"]].copy()
    validos["case_id"] = validos["case_id"].astype(str).str.zfill(3)

    rows = []
    for _, row in tqdm(validos.iterrows(), total=len(validos), desc="Analizando fragmentos"):
        case_id = row["case_id"]
        label_path = LABELS_DIR / f"{case_id}.mha"
        if not label_path.exists():
            print(f"[AVISO] No se encontró label para el caso {case_id}, se omite.")
            continue

        spacing_xyz = (row["spacing_x_mm"], row["spacing_y_mm"], row["spacing_z_mm"])
        rows.append(analizar_caso(case_id, label_path, spacing_xyz))

    df = pd.DataFrame(rows)
    df.to_csv(DETALLE_PATH, index=False)
    print(f"\nDetalle por caso guardado en: {DETALLE_PATH}")

    if (df["n_labels_desconocidos"] > 0).any():
        print("\n[AVISO] Se encontraron labels fuera de la convención oficial (0/1-10/11-20/21-30) en:")
        print(df[df["n_labels_desconocidos"] > 0][["case_id", "n_labels_desconocidos"]].to_string(index=False))

    # -----------------------------------------------------------------
    # GRÁFICO 1 — Histograma de fragmentos totales por caso
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["n_fragmentos_total"], bins=range(1, df["n_fragmentos_total"].max() + 2), edgecolor="black")
    plt.xlabel("Número de fragmentos por caso (total, las 3 regiones)")
    plt.ylabel("Número de casos")
    plt.title("Distribución de fragmentos totales por caso")
    plt.tight_layout()
    plt.savefig(OUTPUT_EDA_DIR / "hist_fragmentos_totales.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # GRÁFICO 2 — Barras de fragmentos promedio por región
    # -----------------------------------------------------------------
    medias_region = df[["n_fragmentos_sacro", "n_fragmentos_coxal_izq", "n_fragmentos_coxal_der"]].mean()
    plt.figure(figsize=(7, 5))
    plt.bar(["Sacro", "Coxal izquierdo", "Coxal derecho"], medias_region.values,
            color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylabel("Promedio de fragmentos por caso")
    plt.title("Fragmentos promedio por región anatómica")
    plt.tight_layout()
    plt.savefig(OUTPUT_EDA_DIR / "barras_fragmentos_por_region.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # GRÁFICO 3 — Boxplot de volumen total de hueso por caso (mm3)
    # -----------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.boxplot(df["volumen_hueso_total_mm3"], vert=True)
    plt.ylabel("Volumen total de hueso segmentado (mm³)")
    plt.title("Distribución del volumen de hueso por caso")
    plt.tight_layout()
    plt.savefig(OUTPUT_EDA_DIR / "boxplot_volumen_hueso_mm3.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # TABLA RESUMEN
    # -----------------------------------------------------------------
    resumen = pd.DataFrame({
        "metrica": [
            "n_casos",
            "fragmentos_total_min", "fragmentos_total_max", "fragmentos_total_media",
            "fragmentos_sacro_media", "fragmentos_coxal_izq_media", "fragmentos_coxal_der_media",
            "casos_sin_fractura_sacro (1 fragmento)",
            "casos_sin_fractura_coxal_izq (1 fragmento)",
            "casos_sin_fractura_coxal_der (1 fragmento)",
            "volumen_hueso_mm3_min", "volumen_hueso_mm3_max", "volumen_hueso_mm3_media",
        ],
        "valor": [
            len(df),
            df["n_fragmentos_total"].min(), df["n_fragmentos_total"].max(), df["n_fragmentos_total"].mean(),
            df["n_fragmentos_sacro"].mean(), df["n_fragmentos_coxal_izq"].mean(), df["n_fragmentos_coxal_der"].mean(),
            (df["n_fragmentos_sacro"] == 1).sum(),
            (df["n_fragmentos_coxal_izq"] == 1).sum(),
            (df["n_fragmentos_coxal_der"] == 1).sum(),
            df["volumen_hueso_total_mm3"].min(), df["volumen_hueso_total_mm3"].max(), df["volumen_hueso_total_mm3"].mean(),
        ]
    })
    resumen.to_csv(OUTPUT_EDA_DIR / "resumen_estadistico.csv", index=False)

    print("\n" + "=" * 60)
    print("RESUMEN EDA DE FRAGMENTOS")
    print("=" * 60)
    print(resumen.to_string(index=False))
    print(f"\nGráficos guardados en: {OUTPUT_EDA_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()