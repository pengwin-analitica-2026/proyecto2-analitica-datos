"""
eda_fragmento_principal_y_tamano.py
-------------------------------------
Dos analisis pendientes de la semana 8, a nivel de FRAGMENTO individual
(no de caso, como eda_fragments.py):

1. Tamano de cada fragmento en mm3 y cuantos cortes axiales ocupa. Sirve
   para estimar cuantos fragmentos pequenos se pierden al bajar la
   resolucion a 224-256 px (advertencia del profesor en la presentacion).

2. Validacion del "fragmento principal": la convencion asumida en el
   proyecto es que la etiqueta que termina en 1 (1 para sacro, 11 para
   coxal izquierdo, 21 para coxal derecho) es siempre el fragmento de
   mayor volumen de su region. Sin esto no se puede calcular la distancia
   de separacion (seccion 3.3 del proyecto), asi que hay que confirmarlo
   contra los 100 casos, no asumirlo.

Convencion de etiquetas (igual que eda_fragments.py):
    0        = fondo
    1-10     = fragmentos de sacro       (1 = supuesto principal)
    11-20    = fragmentos de coxal izq.  (11 = supuesto principal)
    21-30    = fragmentos de coxal der.  (21 = supuesto principal)

Genera:
- data/eda_fragmentos_individuales.csv     (una fila por fragmento)
- data/eda_fragmento_principal_resumen.csv (un resumen por caso+region)
- outputs/eda/hist_tamano_fragmentos_mm3.png
- outputs/eda/hist_cortes_por_fragmento.png

Requisitos:
    pip install SimpleITK numpy pandas matplotlib scipy tqdm

Uso (desde la raiz del repo):
    python src/eda_fragmento_principal_y_tamano.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
MANIFEST_PATH = Path("data/manifest.csv")
LABELS_DIR = Path("data/raw") / "PENGWIN_CT_train_labels"

OUTPUT_EDA_DIR = Path("outputs/eda")
OUTPUT_EDA_DIR.mkdir(parents=True, exist_ok=True)

DETALLE_PATH = Path("data/eda_fragmentos_individuales.csv")
RESUMEN_PATH = Path("data/eda_fragmento_principal_resumen.csv")

# etiqueta "principal" esperada por region, segun la convencion asumida
LABEL_PRINCIPAL_ESPERADO = {"sacro": 1, "coxal_izquierdo": 11, "coxal_derecho": 21}


def region_de_label(label_val: int) -> str:
    """Misma clasificacion que eda_fragments.py."""
    if 1 <= label_val <= 10:
        return "sacro"
    elif 11 <= label_val <= 20:
        return "coxal_izquierdo"
    elif 21 <= label_val <= 30:
        return "coxal_derecho"
    else:
        return "desconocido"


def analizar_fragmentos_caso(case_id: str, label_path: Path, spacing_xyz: tuple) -> list[dict]:
    """
    Devuelve una fila por fragmento (label != 0) presente en el caso, con
    su volumen en mm3 y el numero de cortes axiales (eje z) que ocupa.
    """
    label_img = sitk.ReadImage(str(label_path))
    label_arr = sitk.GetArrayFromImage(label_img)  # (z, y, x)

    sx, sy, sz = spacing_xyz
    voxel_volume_mm3 = sx * sy * sz

    filas = []
    unique_vals = np.unique(label_arr)
    unique_vals = unique_vals[unique_vals != 0]

    for val in unique_vals:
        val = int(val)
        fragmento_mask = label_arr == val
        n_voxeles = int(fragmento_mask.sum())

        # cortes axiales (z) donde el fragmento tiene al menos 1 voxel
        cortes_con_fragmento = np.where(fragmento_mask.any(axis=(1, 2)))[0]
        n_cortes = int(len(cortes_con_fragmento))

        filas.append({
            "case_id": case_id,
            "label": val,
            "region": region_de_label(val),
            "n_voxeles": n_voxeles,
            "volumen_mm3": n_voxeles * voxel_volume_mm3,
            "n_cortes_axiales": n_cortes,
        })

    return filas


def validar_fragmento_principal(df_fragmentos: pd.DataFrame) -> pd.DataFrame:
    """
    Por cada (case_id, region) con al menos un fragmento, determina cual
    label tiene el mayor volumen y compara contra la etiqueta principal
    esperada (1 / 11 / 21).
    """
    filas = []
    for (case_id, region), grupo in df_fragmentos.groupby(["case_id", "region"]):
        if region == "desconocido":
            continue

        label_esperado = LABEL_PRINCIPAL_ESPERADO[region]
        fila_mas_grande = grupo.loc[grupo["volumen_mm3"].idxmax()]
        label_real_mas_grande = int(fila_mas_grande["label"])

        filas.append({
            "case_id": case_id,
            "region": region,
            "n_fragmentos_region": len(grupo),
            "label_principal_esperado": label_esperado,
            "label_con_mayor_volumen": label_real_mas_grande,
            "principal_correcto": label_esperado == label_real_mas_grande,
            "existe_label_principal": (grupo["label"] == label_esperado).any(),
            "volumen_mayor_mm3": fila_mas_grande["volumen_mm3"],
        })

    return pd.DataFrame(filas)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    validos = manifest[manifest["valido"]].copy()
    validos["case_id"] = validos["case_id"].astype(str).str.zfill(3)

    todas_filas = []
    for _, row in tqdm(validos.iterrows(), total=len(validos), desc="Analizando fragmentos"):
        case_id = row["case_id"]
        label_path = LABELS_DIR / f"{case_id}.mha"
        if not label_path.exists():
            print(f"[AVISO] No se encontro label para el caso {case_id}, se omite.")
            continue

        spacing_xyz = (row["spacing_x_mm"], row["spacing_y_mm"], row["spacing_z_mm"])
        todas_filas.extend(analizar_fragmentos_caso(case_id, label_path, spacing_xyz))

    df = pd.DataFrame(todas_filas)
    df.to_csv(DETALLE_PATH, index=False)
    print(f"\nDetalle por fragmento guardado en: {DETALLE_PATH} ({len(df)} fragmentos)")

    if df.empty:
        print("[AVISO] No se encontro ningun fragmento en los casos validos. "
              "Revisa LABELS_DIR y el manifest antes de continuar.")
        return

    # -----------------------------------------------------------------
    # VALIDACION DEL FRAGMENTO PRINCIPAL
    # -----------------------------------------------------------------
    resumen = validar_fragmento_principal(df)
    resumen.to_csv(RESUMEN_PATH, index=False)

    n_total = len(resumen)
    n_correcto = int(resumen["principal_correcto"].sum())
    n_incorrecto = n_total - n_correcto

    print("\n" + "=" * 70)
    print("VALIDACION: ¿la etiqueta 1/11/21 es siempre el fragmento mas grande?")
    print("=" * 70)
    print(f"Casos-region evaluados : {n_total}")
    print(f"Cumple la convencion   : {n_correcto} ({100 * n_correcto / n_total:.1f}%)")
    print(f"NO cumple              : {n_incorrecto} ({100 * n_incorrecto / n_total:.1f}%)")

    if n_incorrecto > 0:
        print("\nCasos donde el fragmento mas grande NO es la etiqueta esperada:")
        excepciones = resumen[~resumen["principal_correcto"]][
            ["case_id", "region", "label_principal_esperado", "label_con_mayor_volumen",
             "existe_label_principal", "n_fragmentos_region"]
        ]
        print(excepciones.to_string(index=False))
        print(
            "\n[IMPORTANTE] Para el calculo de distancia de separacion (seccion 3.3), "
            "el 'fragmento principal' de cada hueso debe definirse como el fragmento "
            "de MAYOR VOLUMEN de su region (columna label_con_mayor_volumen), "
            "NO asumir siempre la etiqueta 1/11/21 si hay excepciones arriba."
        )
    else:
        print(
            "\nLa convencion se cumple en los 100 casos: la etiqueta 1/11/21 "
            "siempre corresponde al fragmento de mayor volumen de su region."
        )

    # -----------------------------------------------------------------
    # GRAFICO 1 — Tamano de fragmentos en mm3 (escala log, hay mucha
    # diferencia de tamano entre el hueso principal y los conminutos)
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["volumen_mm3"], bins=50, edgecolor="black")
    plt.xscale("log")
    plt.xlabel("Volumen del fragmento (mm³, escala log)")
    plt.ylabel("Numero de fragmentos")
    plt.title("Distribucion del tamano de fragmentos individuales (todos los casos)")
    plt.tight_layout()
    plt.savefig(OUTPUT_EDA_DIR / "hist_tamano_fragmentos_mm3.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # GRAFICO 2 — Cuantos cortes axiales ocupa cada fragmento (relevante
    # para saber que tan facil es que un fragmento chico "desaparezca"
    # entre cortes si se pierde resolucion o hay corte-a-corte ruidoso)
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["n_cortes_axiales"], bins=range(1, df["n_cortes_axiales"].max() + 2),
              edgecolor="black")
    plt.xlabel("Numero de cortes axiales que ocupa el fragmento")
    plt.ylabel("Numero de fragmentos")
    plt.title("Extension axial de los fragmentos individuales")
    plt.tight_layout()
    plt.savefig(OUTPUT_EDA_DIR / "hist_cortes_por_fragmento.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # UMBRAL DE RIESGO: fragmentos muy pequenos (< 5 cortes o < 500 mm3,
    # ~ el tamano de un cubo de 8mm de lado) que se pueden perder facil
    # al hacer resize a 224-256 px o con augmentations agresivas.
    # -----------------------------------------------------------------
    pequenos = df[(df["n_cortes_axiales"] < 5) | (df["volumen_mm3"] < 500)]
    print(f"\nFragmentos pequenos (<5 cortes o <500 mm3): {len(pequenos)} de {len(df)} "
          f"({100 * len(pequenos) / len(df):.1f}%) — vigilar en el resize a 224-256 px.")

    print(f"\nGraficos guardados en: {OUTPUT_EDA_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
