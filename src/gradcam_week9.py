"""Grad-CAM (y Smooth Grad-CAM) del detector de semana 9, con y sin atencion.

Pedido de la asesoria: ver donde concentra la "energia" el modelo cuando decide
que hay una region, comparando el modelo con CBAM contra el de control sin CBAM.

Como se calcula (Grad-CAM, Selvaraju et al. 2017; HiResCAM, Draelos y Carin 2020):
  1. Se toma un mapa de caracteristicas del backbone: por defecto el de /8
     (32x32, salida del bloque 3 con su atencion si la tiene); --capa 3 usa el
     de /16 que entra a la cabeza.
  2. Objetivo: para cada region (sacro, coxal izq., coxal der.) se elige la
     celda del grid con mayor puntaje de esa clase y se usa la suma de sus
     logits de objectness y de clase. Asi el mapa responde "por que el modelo
     dice que AQUI hay ESTA region", y no el objectness de todo el corte, que
     esta dominado por las ~250 celdas de fondo.
  3. Grad-CAM clasico: peso de cada canal = promedio espacial de su gradiente;
     CAM = ReLU de la suma ponderada de canales. HiResCAM (por defecto): CAM =
     ReLU de la suma de gradiente x activacion punto a punto. En este detector
     el clasico resalta el fondo (el promedio mezcla canales globales) y
     HiResCAM conserva la localizacion; --metodo gradcam permite comparar.
     El mapa se interpola a 256x256 y se normaliza a [0,1].
  Smooth Grad-CAM: promedio del CAM sobre N copias de la imagen con ruido
  gaussiano (sigma = 0.05 en la escala [0,1]); reduce el ruido del mapa.

Medida cuantitativa ("concentracion de energia"): fraccion de la suma del CAM
que cae dentro de la caja real de esa region. Se divide por la fraccion del
area que ocupa la caja, para que un mapa uniforme valga 1: mayor que 1 es que
el modelo mira la region mas que el resto de la imagen.

Uso (desde la raiz del repo):
    python src/gradcam_week9.py --modelos "con CBAM=outputs/week9_det_cbam32.pth" \\
        "sin CBAM=outputs/week9_det_nocbam32.pth" --casos 024 014 --smooth 8
Salidas: <output-dir>/week9_gradcam_<caso>.png y week9_gradcam_energia.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week9_detection import (REGION_NAMES, boxes_from_label, decode_predictions,
                             detection_model_from_checkpoint)

COLORES = ("#00BFFF", "#00C853", "#FF5252")


def _backbone_con_captura(backbone, x: torch.Tensor, capa: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Recorre el backbone igual que su forward y devuelve (salida, mapa capturado).

    capa = indice del bloque (0..3) cuya salida, ya con su atencion si la
    tiene, se usa para el CAM: 2 -> mapa /8 (32x32), 3 -> mapa /16 (16x16).
    """
    capturado = None
    for i, block in enumerate(backbone.features):
        x = block(x)
        if str(i) in backbone.cbam_intermedios:
            x = backbone._atencion(i, backbone.cbam_intermedios[str(i)], x)
        if i == 3:
            x = backbone._atencion(3, backbone.cbam, x) if backbone.use_cbam else backbone.cbam(x)
        if i == capa:
            x.retain_grad()
            capturado = x
    return x, capturado


def gradcam(model: torch.nn.Module, image: torch.Tensor, clase: int, smooth: int = 0,
            sigma: float = 0.05, seed: int = 0, metodo: str = "hirescam",
            capa: int = 2) -> tuple[np.ndarray, tuple[int, int], float]:
    """Mapa [H,W] en [0,1] para la clase pedida, celda elegida y su puntaje.

    metodo="gradcam": Grad-CAM clasico, peso por canal = promedio espacial del
    gradiente. metodo="hirescam": HiResCAM (Draelos y Carin, 2020), gradiente
    por activacion punto a punto, sin promediar. En una cabeza densa como la
    del grid el promedio mezcla los gradientes de una celda con canales que se
    activan en todo el corte (por ejemplo, "esto es fondo"), y el mapa clasico
    termina resaltando el fondo; HiResCAM conserva la localizacion.
    image: [1,1,H,W]. Si smooth > 0 promedia sobre `smooth` copias con ruido.
    """
    model.eval()
    with torch.no_grad():
        out = model(image)[0]                                    # [gh,gw,5+C]
        score = torch.sigmoid(out[..., 0]) * torch.softmax(out[..., 5:], dim=-1)[..., clase]
        idx = int(score.argmax())
        gi, gj = divmod(idx, out.shape[1])
        puntaje = float(score[gi, gj])
    gen = torch.Generator().manual_seed(seed)
    entradas = [image] if smooth <= 0 else [
        (image + sigma * torch.randn(image.shape, generator=gen)).clamp(0, 1) for _ in range(smooth)]
    acumulado = torch.zeros(image.shape[-2:])
    for x in entradas:
        with torch.enable_grad():
            feats, mapa = _backbone_con_captura(model.backbone, x, capa)
            pred = model.detection_head(feats)[0]
            objetivo = pred[gi, gj, 0] + pred[gi, gj, 5 + clase]
            model.zero_grad(set_to_none=True)
            objetivo.backward()
        grad, act = mapa.grad[0], mapa[0].detach()
        if metodo == "gradcam":
            cam = F.relu((grad.mean(dim=(1, 2))[:, None, None] * act).sum(0))
        else:
            cam = F.relu((grad * act).sum(0))
        cam = F.interpolate(cam[None, None], size=image.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]
        acumulado += cam
    cam = acumulado / len(entradas)
    cam = cam / cam.max() if float(cam.max()) > 0 else cam
    return cam.numpy(), (gi, gj), puntaje


def concentracion(cam: np.ndarray, caja: list[float]) -> tuple[float, float]:
    """(fraccion de energia dentro de la caja, fraccion / area relativa)."""
    h, w = cam.shape
    x0, y0, x1, y1 = (int(round(caja[0] * w)), int(round(caja[1] * h)),
                      int(round(caja[2] * w)), int(round(caja[3] * h)))
    total = float(cam.sum())
    if total <= 0 or x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    dentro = float(cam[y0:y1, x0:x1].sum()) / total
    area = (x1 - x0) * (y1 - y0) / (h * w)
    return dentro, dentro / area


def cargar(path: str) -> torch.nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return detection_model_from_checkpoint(ck).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/week9_detection_slices.npz")
    ap.add_argument("--modelos", nargs="+", required=True, help='"nombre=ruta.pth" ...')
    ap.add_argument("--casos", nargs="*", default=["024", "014", "067", "095"],
                    help="casos de validacion para las figuras")
    ap.add_argument("--smooth", type=int, default=0, help="N copias con ruido (Smooth Grad-CAM)")
    ap.add_argument("--metodo", choices=("hirescam", "gradcam"), default="hirescam")
    ap.add_argument("--capa", type=int, choices=(2, 3), default=2,
                    help="2 = mapa /8 (32x32), 3 = mapa /16 que entra a la cabeza")
    ap.add_argument("--output-dir", default="outputs/figures")
    ap.add_argument("--solo-figuras", action="store_true",
                    help="omite la medida de energia sobre toda la validacion")
    args = ap.parse_args()

    modelos = {}
    for item in args.modelos:
        nombre, ruta = item.split("=", 1)
        modelos[nombre] = cargar(ruta)
    data = np.load(args.cache)
    ids, splits = data["case_ids"], data["split_names"]
    val_idx = np.flatnonzero(splits == "val")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    nombre_metodo = "HiResCAM" if args.metodo == "hirescam" else "Grad-CAM"
    tipo = (f"Smooth {nombre_metodo} (N={args.smooth})" if args.smooth > 0 else nombre_metodo) \
        + f", mapa /{8 if args.capa == 2 else 16}"

    sufijo = f"_{args.metodo}" if args.metodo != "hirescam" else ""
    # 1) concentracion de energia en TODOS los cortes de validacion
    filas = []
    for i in ([] if args.solo_figuras else val_idx):
        img = torch.from_numpy(data["images"][i:i + 1].astype(np.float32))
        gt = boxes_from_label(torch.from_numpy(data["labels"][i:i + 1].astype(np.int64)))[0]
        for g in gt:
            c = int(g["class_id"])
            for nombre, model in modelos.items():
                cam, _, puntaje = gradcam(model, img, c, smooth=args.smooth, metodo=args.metodo, capa=args.capa)
                dentro, ratio = concentracion(cam, g["box"])
                filas.append({"caso": str(ids[i]), "indice": int(i), "region": REGION_NAMES[c],
                              "modelo": nombre, "energia_en_caja": dentro,
                              "concentracion": ratio, "puntaje": puntaje})
    df = pd.DataFrame(filas)
    if not df.empty:
        df.to_csv(out / f"week9_gradcam_energia{sufijo}.csv", index=False)
        resumen_energia(df, tipo, len(val_idx))
    figuras(args, modelos, data, ids, val_idx, out, tipo, sufijo)


def resumen_energia(df: pd.DataFrame, tipo: str, n_cortes: int) -> None:
    tabla = df.pivot_table(index="region", columns="modelo", values="energia_en_caja", aggfunc="mean")
    tabla.loc["todas"] = df.groupby("modelo")["energia_en_caja"].mean()
    conc = df.groupby("modelo")["concentracion"].mean()
    print(f"{tipo}: fraccion de la energia dentro de la caja real ({n_cortes} cortes de validacion)")
    print(tabla.round(3).to_string())
    print("concentracion media (1 = mapa uniforme):", {k: round(float(v), 2) for k, v in conc.items()})


def figuras(args, modelos, data, ids, val_idx, out, tipo, sufijo) -> None:
    # 2) figuras de los casos pedidos (corte central del cache, como el visualizador)
    for cid in args.casos:
        cand = val_idx[ids[val_idx] == str(cid).zfill(3)]
        if len(cand) == 0:
            print(f"caso {cid}: no esta en validacion, se omite"); continue
        i = int(cand[len(cand) // 2])
        img = torch.from_numpy(data["images"][i:i + 1].astype(np.float32))
        gt = {int(g["class_id"]): g["box"] for g in
              boxes_from_label(torch.from_numpy(data["labels"][i:i + 1].astype(np.int64)))[0]}
        fig, axes = plt.subplots(len(modelos), 4, figsize=(16, 4.1 * len(modelos)), squeeze=False)
        for r, (nombre, model) in enumerate(modelos.items()):
            ax = axes[r, 0]
            ax.imshow(img[0, 0], cmap="gray", vmin=0, vmax=1)
            with torch.no_grad():
                pred = decode_predictions(model(img), confidence=0.2, max_por_clase=1)[0]
            for p in pred:
                x0, y0, x1, y1 = [v * 256 for v in p["box"]]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1.5,
                                       edgecolor=COLORES[int(p["class_id"])]))
            ax.set_title(f"{nombre}: predicción (discontinua)", fontsize=10); ax.axis("off")
            for c in range(3):
                ax = axes[r, c + 1]
                ax.imshow(img[0, 0], cmap="gray", vmin=0, vmax=1)
                if c not in gt:
                    ax.set_title(f"{REGION_NAMES[c]}: no está en el corte", fontsize=9); ax.axis("off")
                    continue
                cam, (gi, gj), puntaje = gradcam(model, img, c, smooth=args.smooth,
                                                 metodo=args.metodo, capa=args.capa)
                ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                x0, y0, x1, y1 = [v * 256 for v in gt[c]]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.5,
                                       edgecolor=COLORES[c]))
                ax.plot((gj + 0.5) * 16, (gi + 0.5) * 16, "w+", ms=10, mew=2)
                dentro, _ = concentracion(cam, gt[c])
                ax.set_title(f"{REGION_NAMES[c]} (celda +, p={puntaje:.2f})\n"
                             f"energía en la caja real: {100 * dentro:.0f} %", fontsize=9)
                ax.axis("off")
        fig.suptitle(f"{tipo} — caso {cid}, corte {int(data['z_indices'][i])} (validación). "
                     "Caja continua = región real; rojo = donde mira el modelo", fontsize=12)
        fig.tight_layout()
        ruta = out / f"week9_gradcam_{str(cid).zfill(3)}{sufijo}.png"
        fig.savefig(ruta, dpi=90, bbox_inches="tight"); plt.close(fig)
        print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
