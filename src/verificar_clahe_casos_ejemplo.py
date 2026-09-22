"""
verificar_clahe_casos_ejemplo.py
-----------------------------------
Verificación visual de window_hu() con CLAHE integrado (pengwin_io.py),
sobre los 3 casos de referencia usados en todo el proyecto: 001 (normal),
025 (histograma óseo corrido, el que motivó el cambio) y 068 (padding
extremo, HU mín -6211).

Cada fila: caso. Columnas: sin CLAHE (aplicar_clahe=False) vs con CLAHE
(default, lo que va a usar el pipeline real de aquí en adelante).

Uso (desde la raíz del repo):
    python src/verificar_clahe_casos_ejemplo.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pengwin_io import load_case, window_hu  # noqa: E402

OUTPUT_PATH = Path("outputs/eda/verificacion_clahe_casos_ejemplo.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CASOS = ["001", "025", "068"]


def corte_central_con_hueso(caso_id: str) -> np.ndarray:
    """Corte axial con más vóxeles de hueso (más informativo que el central geométrico)."""
    caso = load_case(caso_id)
    if caso.label is not None:
        vox_por_corte = (caso.label > 0).reshape(caso.label.shape[0], -1).sum(axis=1)
        idx = int(vox_por_corte.argmax())
    else:
        idx = caso.image.shape[0] // 2
    return caso.image[idx]


def main():
    fig, axes = plt.subplots(len(CASOS), 2, figsize=(9, 4 * len(CASOS)))

    for fila, case_id in enumerate(CASOS):
        print(f"Procesando caso {case_id}...")
        hu_slice = corte_central_con_hueso(case_id)

        sin_clahe = window_hu(hu_slice, aplicar_clahe=False)
        con_clahe = window_hu(hu_slice, aplicar_clahe=True)

        axes[fila, 0].imshow(sin_clahe, cmap="gray", vmin=0, vmax=1)
        axes[fila, 0].set_title(f"caso {case_id} — sin CLAHE (antes)")
        axes[fila, 0].axis("off")

        axes[fila, 1].imshow(con_clahe, cmap="gray", vmin=0, vmax=1)
        axes[fila, 1].set_title(f"caso {case_id} — con CLAHE (ahora, default)")
        axes[fila, 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=140)
    print(f"\nGuardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
