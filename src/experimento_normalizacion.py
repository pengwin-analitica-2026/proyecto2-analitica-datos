"""
experimento_normalizacion.py
----------------------------
Barrido de configuraciones de normalizacion para elegir, CON EVIDENCIA, la
que entra al modelo. Responde a la asesoria: el profesor senalo que nuestras
imagenes salen granuladas y con la distribucion de grises mal repartida, y
pidio justificar el numero de bins y los demas parametros de CLAHE.

Metodo evaluado: NGC-CLAHE (normalized gamma-corrected CLAHE), del paper que
recomendo el profesor:
    Rao et al., "An innovative technique for contrast enhancement of computed
    tomography images using normalized gamma-corrected contrast-limited
    adaptive histogram equalization", EURASIP J. Adv. Signal Process. (2015).
    https://doi.org/10.1186/s13634-015-0214-1

Antes de CLAHE se aplica una correccion gamma, que comprime el tejido blando
para que CLAHE no tenga que estirarlo (y de paso no amplifique su ruido).

    G(I)   = I ** gamma                     sobre la ventana osea ya en [0,1]
    salida = CLAHE(G(I), clip_limit, kernel_size, nbins)

A diferencia del paper, no se renormaliza por minimo y maximo de cada imagen:
la ventana ya fija los extremos en 0 y 1, y renormalizar haria que el brillo
dependa del contenido del corte (ver docstring de pengwin_io.window_hu). El
paso gamma es la propia window_hu de produccion.

RESULTADO (corrida --completo, 5 casos): gamma solo, sin CLAHE, gano en todos
los casos; CLAHE no aporto con ningun gamma. Se adopto gamma = 1.5 en
pengwin_io.window_hu. El CNR crece de forma monotona con gamma, asi que el
valor se fijo con un segundo criterio (brillo del hueso); ver comparar_gamma.py.

QUE MIDE (sobre la imagen resultante, usando la mascara real de etiquetas):

  iqr_blando   rango intercuartil del tejido blando. En una ventana osea el
               tejido blando debe ocupar una banda ESTRECHA de grises.
               Mas bajo = mejor.
  sd_blando    desviacion estandar local (ventana 5x5) dentro del tejido
               blando. Es el "granulado" que vio el profesor.
               Mas bajo = mejor.
  contraste    media(hueso) - media(tejido blando). Mas alto = mejor.
  CNR          contraste / sd_blando. Relacion contraste-ruido: resume las
               tres anteriores en un solo numero. ES LA METRICA DE DECISION.
               Mas alto = mejor.

La fila "ventaneo" (sin gamma ni CLAHE) es la REFERENCIA: es lo que se parece
a la imagen que el profesor mostro como objetivo. Cualquier configuracion que
quede por debajo de esa referencia en CNR esta empeorando las cosas.

USO (desde la raiz del repo, con data/raw descargado):

    python src/experimento_normalizacion.py                  # barrido corto
    python src/experimento_normalizacion.py --completo       # barrido amplio
    python src/experimento_normalizacion.py --casos 001 025 033 068

SALIDAS:
    data/experimento_normalizacion.csv          una fila por caso y variante
    data/experimento_normalizacion_resumen.csv  promedio por variante
    outputs/eda/experimento_normalizacion.png   comparacion visual
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import exposure
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pengwin_io import load_case, window_hu  # noqa: E402

CSV_DETALLE = Path("data/experimento_normalizacion.csv")
CSV_RESUMEN = Path("data/experimento_normalizacion_resumen.csv")
FIGURA = Path("outputs/eda/experimento_normalizacion.png")

# Casos por defecto: 001 normal, 025 HU corrido, 033 el que el profesor vio
# granulado, 068 padding negativo, 080 metal extremo.
CASOS_DEFECTO = ["001", "025", "033", "068", "080"]

# Umbral para separar tejido blando de hueso al MEDIR (sobre HU crudo).
HU_BLANDO_MAX = 150
HU_CUERPO = -300


# ---------------------------------------------------------------------------
# Preprocesamiento evaluado
# ---------------------------------------------------------------------------
def aplicar(hu_slice: np.ndarray, gamma: float, clip_limit: float | None,
            tiles: int, nbins: int) -> np.ndarray:
    """Ventaneo + gamma (la MISMA funcion de produccion) y, opcional, CLAHE.

    Usa pengwin_io.window_hu para que lo que se mide aqui sea exactamente lo
    que entra al modelo. La correccion gamma es una potencia fija sobre la
    ventana [0,1], sin renormalizar por imagen (ver docstring de window_hu).
    clip_limit=None omite CLAHE.
    """
    img = window_hu(hu_slice, gamma=gamma)
    if clip_limit is None:
        return img
    if img.max() - img.min() < 1e-8:
        return img
    k = tuple(max(s // tiles, 8) for s in img.shape)   # tejas: shape // tiles
    return exposure.equalize_adapthist(
        img, kernel_size=k, clip_limit=clip_limit, nbins=nbins
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def mascaras(hu_slice: np.ndarray, label_slice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (hueso, tejido_blando) en 2D, definidas sobre HU crudo.

    hueso: la etiqueta validada por ortopedistas, no un umbral.
    blando: dentro del cuerpo, fuera del hueso y por debajo de 150 HU, para
    que no se cuele hueso sin etiquetar ni la camilla.
    """
    cuerpo = ndi.binary_fill_holes(hu_slice > HU_CUERPO)
    etiquetas, n = ndi.label(cuerpo)
    if n > 1:                                   # el paciente = componente mayor
        tam = ndi.sum(cuerpo, etiquetas, index=np.arange(1, n + 1))
        cuerpo = etiquetas == (int(np.argmax(tam)) + 1)
    hueso = label_slice > 0
    blando = cuerpo & ~hueso & (hu_slice < HU_BLANDO_MAX)
    return hueso, blando


def sd_local(img: np.ndarray, mascara: np.ndarray, size: int = 5) -> float:
    """Desviacion estandar local media dentro de la mascara: el 'granulado'."""
    med = ndi.uniform_filter(img, size=size)
    var = ndi.uniform_filter(img * img, size=size) - med * med
    return float(np.sqrt(np.clip(var, 0, None))[mascara].mean())


def medir(img: np.ndarray, hueso: np.ndarray, blando: np.ndarray) -> dict:
    if hueso.sum() < 50 or blando.sum() < 50:
        return {}
    q25, q75 = np.percentile(img[blando], [25, 75])
    iqr = float(q75 - q25)
    sd = sd_local(img, blando)
    contraste = float(img[hueso].mean() - img[blando].mean())
    return {
        "iqr_blando": iqr,
        "sd_blando": sd,
        "contraste": contraste,
        "cnr": contraste / sd if sd > 1e-6 else np.nan,
        "media_hueso": float(img[hueso].mean()),
        "media_blando": float(img[blando].mean()),
    }


# ---------------------------------------------------------------------------
# Barrido
# ---------------------------------------------------------------------------
def construir_variantes(completo: bool) -> list[dict]:
    """(nombre, gamma, clip_limit, tiles, nbins). clip_limit=None -> sin CLAHE."""
    v = [
        # Referencia: lo que el profesor mostro como objetivo.
        {"nombre": "ventaneo (referencia)", "gamma": 1.0, "clip": None,
         "tiles": 0, "nbins": 0},
        # Lo que tenemos hoy en produccion.
        {"nombre": "CLAHE actual", "gamma": 1.0, "clip": 0.02,
         "tiles": 8, "nbins": 256},
    ]
    if completo:
        gammas = [1.0, 1.2, 1.5, 1.8, 2.2]
        clips = [0.002, 0.004, 0.008, 0.015]
        tiles_l = [8, 16]
        nbins_l = [32, 64, 128, 256]
    else:
        gammas = [1.0, 1.5, 2.0]
        clips = [0.003, 0.008]
        tiles_l = [8, 16]
        nbins_l = [64, 256]
    for g, c, t, nb in product(gammas, clips, tiles_l, nbins_l):
        v.append({"nombre": f"NGC g={g} clip={c} tejas={t} bins={nb}",
                  "gamma": g, "clip": c, "tiles": t, "nbins": nb})
    # Gamma solo, sin CLAHE: util para saber cuanto aporta cada parte.
    for g in ([1.5, 2.0] if not completo else [1.2, 1.5, 1.8, 2.2]):
        v.append({"nombre": f"solo gamma g={g}", "gamma": g, "clip": None,
                  "tiles": 0, "nbins": 0})
    return v


def corte_con_mas_hueso(caso) -> int:
    if caso.label is None:
        raise RuntimeError("el caso no tiene mascara")
    por_corte = (caso.label > 0).reshape(caso.label.shape[0], -1).sum(axis=1)
    return int(por_corte.argmax())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", nargs="+", default=CASOS_DEFECTO)
    ap.add_argument("--completo", action="store_true",
                    help="barrido amplio (mas lento, ~5x variantes)")
    ap.add_argument("--top", type=int, default=6,
                    help="cuantas variantes se dibujan en la figura")
    args = ap.parse_args()

    variantes = construir_variantes(args.completo)
    print(f"{len(variantes)} variantes x {len(args.casos)} casos "
          f"= {len(variantes) * len(args.casos)} evaluaciones\n")

    filas, cortes = [], {}
    for case_id in args.casos:
        caso = load_case(case_id, load_label=True)
        z = corte_con_mas_hueso(caso)
        hu = caso.image[z].astype(np.float32)
        lb = caso.label[z]
        hueso, blando = mascaras(hu, lb)
        cortes[case_id] = (hu, hueso, blando, z)
        for var in tqdm(variantes, desc=f"caso {case_id} (corte {z})"):
            img = aplicar(hu, var["gamma"], var["clip"], var["tiles"], var["nbins"])
            m = medir(img, hueso, blando)
            if m:
                filas.append({"caso": case_id, "corte": z, **{
                    k: var[k] for k in ("nombre", "gamma", "clip", "tiles", "nbins")
                }, **m})

    df = pd.DataFrame(filas)
    CSV_DETALLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_DETALLE, index=False)

    resumen = (df.groupby(["nombre", "gamma", "clip", "tiles", "nbins"], dropna=False)
                 [["iqr_blando", "sd_blando", "contraste", "cnr"]]
                 .mean().reset_index()
                 .sort_values("cnr", ascending=False))
    resumen.to_csv(CSV_RESUMEN, index=False)

    ref = resumen[resumen["nombre"] == "ventaneo (referencia)"].iloc[0]
    act = resumen[resumen["nombre"] == "CLAHE actual"].iloc[0]

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n" + "=" * 96)
    print("RESUMEN — promedio sobre los casos, ordenado por CNR (mayor = mejor)")
    print("=" * 96)
    print(resumen.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "-" * 96)
    print(f"REFERENCIA (ventaneo solo) : CNR={ref['cnr']:.3f}  "
          f"iqr_blando={ref['iqr_blando']:.3f}  sd_blando={ref['sd_blando']:.4f}")
    print(f"HOY EN PRODUCCION          : CNR={act['cnr']:.3f}  "
          f"iqr_blando={act['iqr_blando']:.3f}  sd_blando={act['sd_blando']:.4f}")
    mejor = resumen.iloc[0]
    print(f"MEJOR DEL BARRIDO          : CNR={mejor['cnr']:.3f}  "
          f"iqr_blando={mejor['iqr_blando']:.3f}  sd_blando={mejor['sd_blando']:.4f}")
    print(f"                             -> {mejor['nombre']}")
    print("-" * 96)
    if act["cnr"] < ref["cnr"]:
        print("AVISO: la configuracion actual es PEOR que el ventaneo solo. "
              "Confirma lo senalado en la asesoria.")

    # ---------------- Figura comparativa ----------------
    elegidas = ["ventaneo (referencia)", "CLAHE actual"]
    for nom in resumen["nombre"]:
        if nom not in elegidas:
            elegidas.append(nom)
        if len(elegidas) >= args.top:
            break

    n_f, n_c = len(args.casos), len(elegidas)
    fig, axes = plt.subplots(n_f, n_c, figsize=(3.1 * n_c, 3.3 * n_f), squeeze=False)
    for i, case_id in enumerate(args.casos):
        hu, hueso, blando, z = cortes[case_id]
        for j, nom in enumerate(elegidas):
            v = next(v for v in variantes if v["nombre"] == nom)
            img = aplicar(hu, v["gamma"], v["clip"], v["tiles"], v["nbins"])
            ax = axes[i][j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            fila = df[(df["caso"] == case_id) & (df["nombre"] == nom)]
            cnr = fila["cnr"].iloc[0] if len(fila) else float("nan")
            if i == 0:
                ax.set_title(nom.replace(" ", "\n", 1), fontsize=8.5)
            ax.text(0.02, 0.02, f"caso {case_id} · CNR {cnr:.2f}",
                    transform=ax.transAxes, fontsize=8, color="yellow",
                    va="bottom", ha="left")
    fig.suptitle("Semana 9 — eleccion de la normalizacion (NGC-CLAHE)\n"
                 "columnas ordenadas por CNR; la 1a es el ventaneo sin realce",
                 fontsize=12)
    fig.tight_layout()
    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURA, dpi=150, bbox_inches="tight")

    print(f"\nCSV detalle : {CSV_DETALLE}")
    print(f"CSV resumen : {CSV_RESUMEN}")
    print(f"Figura      : {FIGURA}")


if __name__ == "__main__":
    main()