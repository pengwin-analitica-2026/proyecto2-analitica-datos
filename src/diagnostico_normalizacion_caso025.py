"""
diagnostico_normalizacion_caso025.py
--------------------------------------
Compara 3 métodos de normalización de contraste (gamma, ecualización global,
CLAHE) sobre el caso 025 (histograma óseo corrido ~150-200 HU respecto a lo
típico, ver docs/resumen_semana8.md) y sobre un caso típico (007), para
elegir cuál integrar en pengwin_io.py sin dañar los casos que ya están bien.

Diagnóstico previo (sobre data/eda_intensidades_detalle.csv, sin necesidad
de tocar los volúmenes crudos): el caso 025 es el único con hueso_media y
hueso_p50 estadísticamente atípicos (z < -2) frente a los 100 casos. El
tejido blando y el HU máximo NO son atípicos — descarta la hipótesis de
metal, confirma la de mala distribución de HU específica del hueso.

Criterio de selección (no solo visual): el método debe aplicarse igual a
TODOS los casos dentro de pengwin_io.py. Gamma usa una curva fija (mismo
exponente para todos) — puede arreglar el 025 pero distorsionar casos ya
normales. Ecualización global y CLAHE se adaptan al histograma de CADA
imagen individualmente, así que en un caso ya bien distribuido el efecto es
casi neutro. Por eso se comparan lado a lado: caso 025 (debe mejorar) y
caso típico 007 (NO debe empeorar).

Requisitos:
    pip install scikit-image matplotlib

Uso (desde la raíz del repo):
    python src/diagnostico_normalizacion_caso025.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pengwin_io import load_case, window_hu  # noqa: E402

OUTPUT_PATH = Path("outputs/eda/diagnostico_normalizacion_caso025.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CASOS = {"025": "atípico", "007": "típico"}
GAMMA = 0.7  # <1 aclara las zonas oscuras (donde cayó el hueso corrido)


def corte_central_con_hueso(caso_id: str) -> np.ndarray:
    """Corte axial ventaneado [0,1] con más vóxeles de hueso (más informativo que el central geométrico)."""
    caso = load_case(caso_id)
    ventaneado = window_hu(caso.image)
    if caso.label is not None:
        vox_por_corte = (caso.label > 0).reshape(caso.label.shape[0], -1).sum(axis=1)
        idx = int(vox_por_corte.argmax())
    else:
        idx = ventaneado.shape[0] // 2
    return ventaneado[idx]


def aplicar_metodos(img01: np.ndarray) -> dict:
    return {
        "original (ventaneado)": img01,
        f"gamma (g={GAMMA})": exposure.adjust_gamma(img01, gamma=GAMMA),
        "ecualización global": exposure.equalize_hist(img01),
        "CLAHE": exposure.equalize_adapthist(img01, clip_limit=0.02),
    }


def main():
    fig, axes = plt.subplots(len(CASOS), 4, figsize=(16, 8))

    for fila, (case_id, etiqueta) in enumerate(CASOS.items()):
        print(f"Procesando caso {case_id} ({etiqueta})...")
        corte = corte_central_con_hueso(case_id)
        metodos = aplicar_metodos(corte)

        for col, (nombre, imagen) in enumerate(metodos.items()):
            ax = axes[fila, col]
            ax.imshow(imagen, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{nombre}\ncaso {case_id} ({etiqueta})", fontsize=9)
            ax.axis("off")

            # métrica rápida de contraste: desviación estándar de la imagen
            # (qué tanto se "despliega" el histograma; más alto = más contraste)
            ax.set_xlabel(f"std={imagen.std():.3f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=140)
    print(f"\nComparación guardada en: {OUTPUT_PATH}")
    print(
        "\nRevisar visualmente: el caso 025 (fila superior) debe verse con mejor "
        "contraste óseo tras la corrección, y el caso 007 (fila inferior, ya "
        "normal) NO debe verse sobre-procesado ni perder detalle."
    )


if __name__ == "__main__":
    main()
