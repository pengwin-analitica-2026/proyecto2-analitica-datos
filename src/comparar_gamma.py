"""
comparar_gamma.py
-----------------
Dibuja SOLO la comparacion que hace falta para decidir el gamma, sin rehacer
el barrido completo: ventaneo, el CLAHE que teniamos, y la escalera de gamma.

El barrido (experimento_normalizacion.py) ordena por CNR y por eso su figura
solo muestra los gamma mas altos; el CNR crece de forma monotona con gamma, asi
que el maximo de esa metrica no es la mejor imagen. Aqui se ven lado a lado
para juzgar lo que el CNR no captura: cuanto brillo y cuanta textura interna
conserva el hueso.

USO (desde la raiz del repo):
    python src/comparar_gamma.py
    python src/comparar_gamma.py --gammas 1.2 1.5 2.0 2.2 --casos 001 025 033

SALIDA:
    outputs/eda/comparacion_gamma.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experimento_normalizacion import (aplicar, mascaras, medir,  # noqa: E402
                                       corte_con_mas_hueso, CASOS_DEFECTO)
from pengwin_io import load_case, GAMMA  # noqa: E402

SALIDA = Path("outputs/eda/comparacion_gamma.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", nargs="+", default=CASOS_DEFECTO)
    ap.add_argument("--gammas", nargs="+", type=float, default=[1.2, 1.5, 2.0, 2.2])
    ap.add_argument("--zoom", action="store_true",
                    help="recorta al hueso para ver la textura trabecular")
    args = ap.parse_args()

    columnas = [("ventaneo\n(referencia)", 1.0, None),
                ("CLAHE actual\n(descartado)", 1.0, 0.02)]
    columnas += [(f"gamma {g}" + ("\n(elegido)" if g == GAMMA else ""), g, None)
                 for g in args.gammas]

    n_f, n_c = len(args.casos), len(columnas)
    fig, axes = plt.subplots(n_f, n_c, figsize=(3.2 * n_c, 3.5 * n_f), squeeze=False)

    print(f"{'caso':>6} {'variante':<22} {'CNR':>7} {'brillo hueso':>13} "
          f"{'ruido blando':>13}")
    print("-" * 66)

    for i, case_id in enumerate(args.casos):
        caso = load_case(case_id, load_label=True)
        z = corte_con_mas_hueso(caso)
        hu = caso.image[z].astype(np.float32)
        lb = caso.label[z]
        hueso, blando = mascaras(hu, lb)

        recorte = None
        if args.zoom:
            ys, xs = np.nonzero(hueso)
            m = 20
            recorte = (slice(max(ys.min() - m, 0), ys.max() + m),
                       slice(max(xs.min() - m, 0), xs.max() + m))

        for j, (titulo, g, clip) in enumerate(columnas):
            img = aplicar(hu, g, clip, 8, 256)
            m = medir(img, hueso, blando)
            ax = axes[i][j]
            ax.imshow(img[recorte] if recorte else img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(titulo, fontsize=10)
            ax.text(0.02, 0.02,
                    f"caso {case_id}\nCNR {m['cnr']:.2f} · hueso {m['media_hueso']:.2f}",
                    transform=ax.transAxes, fontsize=8, color="yellow",
                    va="bottom", ha="left", linespacing=1.3)
            print(f"{case_id:>6} {titulo.replace(chr(10), ' '):<22} "
                  f"{m['cnr']:>7.2f} {m['media_hueso']:>13.3f} {m['sd_blando']:>13.4f}")

    fig.suptitle("Eleccion del gamma — el CNR crece siempre con gamma, "
                 "pero el hueso se oscurece\n"
                 "mirar brillo y textura interna del hueso, no solo el CNR",
                 fontsize=12)
    fig.tight_layout()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=150, bbox_inches="tight")
    print(f"\nFigura: {SALIDA}")


if __name__ == "__main__":
    main()