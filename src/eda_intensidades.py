"""
eda_intensidades.py
--------------------
EDA de INTENSIDADES (HU) del dataset PENGWIN. Complementa a eda_fragments.py
(que analiza las máscaras). Caracteriza dos fenómenos del CT crudo:

1. Metal / endurecimiento de haz: vóxeles > HU_METAL por caso, volumen en mm3,
   número de cortes afectados y corte con más metal.
2. Desajuste de histograma entre casos: histograma de HU del cuerpo y del hueso
   (usando la máscara), percentiles, media y desviación por caso.

Genera:
- data/eda_intensidades_detalle.csv        (una fila por caso)
- data/eda_intensidades_histogramas.npz    (histogramas normalizados por caso)

ARCHIVO GENERADO desde notebooks/02_eda_intensidades.ipynb con %%writefile.
Para cambiarlo, editar la celda del notebook y volver a ejecutarla.

Uso (desde la raíz del repo):
    python src/eda_intensidades.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
DATA_ROOT = Path("data/raw")
IMAGES_DIRS = [
    DATA_ROOT / "PENGWIN_CT_train_images_part1",
    DATA_ROOT / "PENGWIN_CT_train_images_part2",
]
LABELS_DIR = DATA_ROOT / "PENGWIN_CT_train_labels"
MANIFEST_PATH = Path("data/manifest.csv")

DETALLE_PATH = Path("data/eda_intensidades_detalle.csv")
HIST_PATH = Path("data/eda_intensidades_histogramas.npz")

CLIP_MIN, CLIP_MAX = -1024, 3000      # mismo clip que el ventaneo
HU_METAL = 3000                       # por encima de esto no hay tejido: es metal
HU_CUERPO = -500                      # por encima de esto: cuerpo (excluye aire y padding)
BIN_HU = 10
HU_EDGES = np.arange(CLIP_MIN, CLIP_MAX + BIN_HU + 1, BIN_HU)
HU_CENTERS = (HU_EDGES[:-1] + HU_EDGES[1:]) / 2


# ---------------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------------
def find_image_path(case_id):
    for img_dir in IMAGES_DIRS:
        candidate = img_dir / f"{case_id}.mha"
        if candidate.exists():
            return candidate
    return None


def load_case(case_id):
    """Devuelve (volumen HU crudo (z,y,x), máscara (z,y,x), spacing (x,y,z) en mm)."""
    img = sitk.ReadImage(str(find_image_path(case_id)))
    lab = sitk.ReadImage(str(LABELS_DIR / f"{case_id}.mha"))
    return sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(lab), img.GetSpacing()


def percentiles_desde_hist(counts, qs):
    """Percentiles aproximados (resolución BIN_HU) a partir de un histograma; evita ordenar millones de vóxeles."""
    total = counts.sum()
    if total == 0:
        return [np.nan] * len(qs)
    cdf = np.cumsum(counts) / total
    return [float(HU_CENTERS[min(np.searchsorted(cdf, q / 100), len(HU_CENTERS) - 1)]) for q in qs]


def analizar_caso(case_id):
    vol, lab, (sx, sy, sz) = load_case(case_id)
    vox_mm3 = sx * sy * sz

    # --- metal (sobre HU crudo, antes del clip) ---
    metal = vol > HU_METAL
    metal_por_corte = metal.reshape(metal.shape[0], -1).sum(axis=1)
    n_metal = int(metal_por_corte.sum())

    # --- padding: valores por debajo del aire ---
    frac_padding = float((vol < CLIP_MIN).mean())

    # --- histogramas (sobre HU con clip) ---
    volc = np.clip(vol, CLIP_MIN, CLIP_MAX)
    m_cuerpo = (volc > HU_CUERPO) & ~metal
    m_hueso = (lab > 0) & ~metal
    h_cuerpo, _ = np.histogram(volc[m_cuerpo], bins=HU_EDGES)
    h_hueso, _ = np.histogram(volc[m_hueso], bins=HU_EDGES)

    c_p1, c_p50, c_p99 = percentiles_desde_hist(h_cuerpo, [1, 50, 99])
    b_p5, b_p50, b_p95 = percentiles_desde_hist(h_hueso, [5, 50, 95])
    v_cuerpo = volc[m_cuerpo].astype(np.float64)
    v_hueso = volc[m_hueso].astype(np.float64)

    row = {
        "case_id": case_id,
        "hu_max_crudo": float(vol.max()),
        "hu_min_crudo": float(vol.min()),
        "frac_padding": frac_padding,
        "n_vox_metal": n_metal,
        "vol_metal_mm3": n_metal * vox_mm3,
        "n_cortes_con_metal": int((metal_por_corte > 0).sum()),
        "corte_max_metal": int(metal_por_corte.argmax()) if n_metal > 0 else -1,
        "metal_dentro_mascara_vox": int((metal & (lab > 0)).sum()),
        "cuerpo_media": float(v_cuerpo.mean()), "cuerpo_std": float(v_cuerpo.std()),
        "cuerpo_p1": c_p1, "cuerpo_p50": c_p50, "cuerpo_p99": c_p99,
        "hueso_media": float(v_hueso.mean()), "hueso_std": float(v_hueso.std()),
        "hueso_p5": b_p5, "hueso_p50": b_p50, "hueso_p95": b_p95,
    }
    # histogramas normalizados (densidad) para poder comparar casos de distinto tamaño
    return row, h_cuerpo / max(h_cuerpo.sum(), 1), h_hueso / max(h_hueso.sum(), 1)


def main():
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"case_id": str})
    casos = manifest.loc[manifest["valido"], "case_id"].str.zfill(3).tolist()

    rows, hc, hb = [], [], []
    for case_id in tqdm(casos, desc="Analizando intensidades"):
        row, h_cuerpo, h_hueso = analizar_caso(case_id)
        rows.append(row); hc.append(h_cuerpo); hb.append(h_hueso)

    df = pd.DataFrame(rows)
    df.to_csv(DETALLE_PATH, index=False)
    np.savez_compressed(HIST_PATH, case_ids=np.array(casos), centers=HU_CENTERS,
                        hist_cuerpo=np.array(hc), hist_hueso=np.array(hb))

    print(f"\nDetalle guardado en      : {DETALLE_PATH}")
    print(f"Histogramas guardados en : {HIST_PATH}")
    print(f"Casos con algún vóxel > {HU_METAL} HU: {(df['n_vox_metal'] > 0).sum()} de {len(df)}")
    print(f"Mediana de HU del hueso entre casos: {df['hueso_p50'].min():.0f} a {df['hueso_p50'].max():.0f} HU")


if __name__ == "__main__":
    main()
