"""Justificacion del preprocesamiento con Otsu contra las mascaras del dataset.

Pedido de la asesoria de semana 9 (profesor Ferro): a cada variante de
preprocesamiento se le aplica un umbral de Otsu dentro del cuerpo, y la
mascara que sale se compara con la mascara osea de PENGWIN (Dice, precision y
recall) en varios casos al azar. La variante cuya mascara concuerda mejor con
el label es la que mejor separa el hueso del resto.

Ademas de Otsu se reporta el Dice con el MEJOR umbral posible para cada
variante ("Dice optimo"). Sirve para separar dos cosas distintas:

  * cuanta informacion de hueso conserva la imagen  -> Dice optimo
  * donde cae el umbral automatico de Otsu           -> Dice Otsu

Ventaneo, gamma 1.5 y gamma 2.0 son funciones monotonas de la misma imagen:
cualquier mascara por umbral de una se puede obtener con algun umbral de la
otra, asi que su Dice optimo es practicamente el mismo y lo que las diferencia
con Otsu es solo la posicion del umbral. CLAHE y la ecualizacion global, en
cambio, cambian los grises segun la zona y si pueden ganar o perder
informacion; ahi el Dice optimo si las distingue.

Limitacion: el label de PENGWIN solo marca la pelvis. El femur, la columna o
la camilla dentro del cuerpo son hueso (o casi) y Otsu los marca, asi que
cuentan como falsos positivos para todas las variantes por igual.

Uso (desde la raiz del repo):
    python src/otsu_preprocesamiento.py --n-casos 5 --cortes-por-caso 5 --seed 42
Salidas: data/otsu_preprocesamiento.csv (una fila por corte y variante),
         <salida-dir>/otsu_preprocesamiento.png y .json (resumen).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import exposure
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experimento_normalizacion import HU_CUERPO, aplicar, mascaras
from pengwin_io import load_case, resize_label_slice, resize_slice, window_hu
from prepare_week9_detection_data import choose_slices


def ecualizacion_global(hu_slice: np.ndarray) -> np.ndarray:
    img = window_hu(hu_slice, gamma=1.0)
    if img.max() - img.min() < 1e-8:
        return img
    return exposure.equalize_hist(img).astype(np.float32)


# nombre -> funcion(hu_slice) -> imagen [0,1]
VARIANTES = {
    "ventaneo": lambda hu: aplicar(hu, 1.0, None, 8, 256),
    "gamma 1.5 (elegido)": lambda hu: aplicar(hu, 1.5, None, 8, 256),
    "gamma 2.0": lambda hu: aplicar(hu, 2.0, None, 8, 256),
    "CLAHE": lambda hu: aplicar(hu, 1.0, 0.02, 8, 256),
    "CLAHE + gamma 1.5": lambda hu: aplicar(hu, 1.5, 0.02, 8, 256),
    "ecualización global": ecualizacion_global,
}


def silueta(hu_slice: np.ndarray) -> np.ndarray:
    """Cuerpo del paciente en 2D: componente mayor de HU > umbral, con huecos
    rellenos (mismo criterio que experimento_normalizacion.mascaras)."""
    cuerpo = ndi.binary_fill_holes(hu_slice > HU_CUERPO)
    etiquetas, n = ndi.label(cuerpo)
    if n > 1:
        tam = ndi.sum(cuerpo, etiquetas, index=np.arange(1, n + 1))
        cuerpo = etiquetas == (int(np.argmax(tam)) + 1)
    return cuerpo


def dice_pr(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = float((pred & gt).sum())
    p, g = float(pred.sum()), float(gt.sum())
    dice = 2 * tp / (p + g) if p + g > 0 else 1.0
    return dice, (tp / p if p else 0.0), (tp / g if g else 0.0)


def dice_optimo(img: np.ndarray, cuerpo: np.ndarray, gt: np.ndarray, pasos: int = 200) -> tuple[float, float]:
    """Mejor Dice alcanzable con un umbral global sobre esta imagen."""
    valores = img[cuerpo]
    umbrales = np.unique(np.quantile(valores, np.linspace(0.0, 1.0, pasos, endpoint=False)))
    mejor, mejor_t = -1.0, float(umbrales[0])
    for t in umbrales:
        d = dice_pr((img > t) & cuerpo, gt)[0]
        if d > mejor:
            mejor, mejor_t = d, float(t)
    return mejor, mejor_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-casos", type=int, default=5)
    ap.add_argument("--cortes-por-caso", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--casos", nargs="*", default=None, help="casos fijos en vez de al azar")
    ap.add_argument("--size", type=int, default=256, help="tamaño de entrada del modelo")
    ap.add_argument("--salida-dir", default="outputs/eda")
    ap.add_argument("--csv", default="data/otsu_preprocesamiento.csv")
    args = ap.parse_args()

    casos = args.casos or sorted(random.Random(args.seed).sample(
        [f"{i:03d}" for i in range(1, 101)], args.n_casos))
    filas, ejemplo = [], None
    for cid in casos:
        caso = load_case(cid)
        zs = choose_slices(caso.label, args.cortes_por_caso)
        for z in zs:
            hu = resize_slice(caso.image[z].astype(np.float32), args.size)
            lab = resize_label_slice(caso.label[z], args.size)
            gt, blando = mascaras(hu, lab)
            cuerpo = silueta(hu)
            # Region "solo pelvis": el cuerpo sin el hueso que el label no marca
            # (femur, columna; > 150 HU fuera de la etiqueta). Aisla la pregunta
            # "separa la pelvis del tejido blando?" del sesgo del label.
            pelvis = gt | blando
            for nombre, f in VARIANTES.items():
                img = f(hu)
                t = threshold_otsu(img[cuerpo]) if cuerpo.any() else 0.5
                pred = (img > t) & cuerpo
                d, p, r = dice_pr(pred, gt)
                d_opt, t_opt = dice_optimo(img, cuerpo, gt)
                tp = threshold_otsu(img[pelvis]) if pelvis.any() else 0.5
                d_pelvis = dice_pr((img > tp) & pelvis, gt)[0]
                filas.append({"caso": cid, "z": int(z), "variante": nombre, "dice": d,
                              "precision": p, "recall": r, "umbral_otsu": float(t),
                              "dice_optimo": d_opt, "umbral_optimo": t_opt,
                              "dice_sin_hueso_no_etiquetado": d_pelvis})
                if ejemplo is None or (cid, z) == ejemplo["clave"]:
                    if ejemplo is None:
                        ejemplo = {"clave": (cid, z), "lab": gt, "paneles": {}}
                    ejemplo["paneles"][nombre] = (img, pred, d)
        print(f"caso {cid}: {len(zs)} cortes evaluados")

    df = pd.DataFrame(filas)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    orden = list(VARIANTES)
    res = (df.groupby("variante")[["dice", "precision", "recall", "dice_optimo",
                                    "dice_sin_hueso_no_etiquetado"]]
             .mean().reindex(orden))
    res["dice_min"] = df.groupby("variante")["dice"].min().reindex(orden)

    print(f"\nCasos: {casos} | {args.cortes_por_caso} cortes por caso | Otsu dentro del cuerpo vs label > 0")
    print(f"{'variante':22s}{'Dice Otsu':>10s}{'precisión':>11s}{'recall':>8s}{'Dice mín':>10s}"
          f"{'Dice óptimo':>13s}{'Otsu sin fémur':>16s}")
    for v, r in res.iterrows():
        print(f"{v:22s}{r['dice']:10.3f}{r['precision']:11.3f}{r['recall']:8.3f}"
              f"{r['dice_min']:10.3f}{r['dice_optimo']:13.3f}{r['dice_sin_hueso_no_etiquetado']:16.3f}")
    def mejores(col: str) -> str:
        # Empates (diferencia < 0.001) se reportan juntos: con umbral optimo las
        # variantes monotonas empatan por construccion.
        top = res[col].max()
        return " = ".join(v for v in orden if top - res.loc[v, col] < 1e-3)
    mejor_otsu, mejor_opt = mejores("dice"), mejores("dice_optimo")
    print(f"\nMejor con Otsu: {mejor_otsu} | mejor con umbral óptimo (empate si hay '='): {mejor_opt}")

    salida = Path(args.salida_dir); salida.mkdir(parents=True, exist_ok=True)
    (salida / "otsu_preprocesamiento.json").write_text(json.dumps({
        "casos": casos, "cortes_por_caso": args.cortes_por_caso, "seed": args.seed,
        "resumen": {v: {k: float(r[k]) for k in res.columns} for v, r in res.iterrows()},
        "mejor_otsu": mejor_otsu, "mejor_optimo": mejor_opt}, indent=2, ensure_ascii=False))

    n = len(orden)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    for j, nombre in enumerate(orden):
        img, pred, d = ejemplo["paneles"][nombre]
        axes[0, j].imshow(img, cmap="gray", vmin=0, vmax=1); axes[0, j].set_title(nombre, fontsize=10)
        axes[1, j].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(np.ma.masked_where(~pred, pred), cmap="autumn", alpha=0.55)
        axes[1, j].contour(ejemplo["lab"], levels=[0.5], colors="lime", linewidths=0.8)
        axes[1, j].set_title(f"Otsu (naranja) vs label (verde)\nDice {d:.2f}", fontsize=9)
        for a in axes[:, j]:
            a.axis("off")
    cid, z = ejemplo["clave"]
    fig.suptitle(f"Otsu sobre cada preprocesamiento — caso {cid}, corte {z} "
                 f"(media de {len(casos)} casos: ver tabla)", fontsize=12)
    fig.tight_layout()
    fig.savefig(salida / "otsu_preprocesamiento.png", dpi=110, bbox_inches="tight")
    print(f"Guardado: {args.csv}, {salida / 'otsu_preprocesamiento.json'} y .png")


if __name__ == "__main__":
    main()