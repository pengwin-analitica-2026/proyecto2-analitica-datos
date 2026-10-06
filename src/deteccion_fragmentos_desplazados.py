"""Que hace el detector de semana 9 con los fragmentos desplazados.

Pedido de la asesoria: el profesor quiere ver casos con fragmentos separados
de su hueso. Acepta que el detector los encierre en la caja del hueso o que les
ponga una caja aparte, siempre que sea de la MISMA clase (sacro, coxal izq. o
coxal der.). No acepta que los ignore ni que les ponga la clase de otro hueso.

Para cada caso del split pedido:
  1. Se miden en 3D, sobre la etiqueta real, las distancias borde a borde de
     cada fragmento secundario a su fragmento principal (fragment_distances,
     con el spacing del header, en mm).
  2. Se toman los fragmentos a mas de --min-dist mm (los --top mas lejanos).
  3. Se elige el corte axial donde el fragmento tiene mas area, se preprocesa
     igual que el cache (window_hu con gamma + 256x256) y se pasa al detector.
  4. Veredicto: la caja predicha que cubre mas pixeles del fragmento.
     misma clase  -> cubre al menos --cobertura del fragmento y es de su hueso
     otra clase   -> lo cubre una caja de otro hueso
     ninguna      -> ninguna caja cubre el fragmento (ignorado)
     Se evalua con todas las cajas tras NMS y con una caja por clase, porque
     limitar a una por clase puede dejar fuera un fragmento lejano que tenia
     su propia caja.
  5. Se anota si el corte cae dentro del rango axial con que se entreno el
     detector (15 %-85 % de la extension osea): fuera de ese rango el
     detector nunca vio cortes asi.

Uso (desde la raiz del repo):
    python src/deteccion_fragmentos_desplazados.py --checkpoint outputs/week9_det_cbam32.pth
Salidas: <output-dir>/resumen.csv y una figura por fragmento.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fragment_distances import REGIONS, measure_fragment_distances
from pengwin_io import load_case, resize_label_slice, resize_slice, window_hu
from week9_detection import REGION_NAMES, decode_predictions, detection_model_from_checkpoint

COLORES = ("#00BFFF", "#00C853", "#FF5252")
RANGO_ENTRENAMIENTO = (0.15, 0.85)


def veredicto(frag: np.ndarray, cajas: list[dict], clase: int, cobertura_min: float) -> dict:
    """Caja que mas cubre el fragmento y su veredicto."""
    h, w = frag.shape
    ys, xs = np.nonzero(frag)
    mejor = {"veredicto": "ninguna", "cobertura": 0.0, "clase_caja": "", "puntaje": 0.0}
    for c in cajas:
        x0, y0, x1, y1 = c["box"]
        dentro = ((xs >= x0 * w) & (xs < x1 * w) & (ys >= y0 * h) & (ys < y1 * h)).mean()
        if dentro > mejor["cobertura"]:
            mejor = {"cobertura": float(dentro), "clase_caja": REGION_NAMES[int(c["class_id"])],
                     "puntaje": float(c["score"]),
                     "veredicto": "misma clase" if int(c["class_id"]) == clase else "otra clase"}
    if mejor["cobertura"] < cobertura_min:
        mejor["veredicto"] = "ninguna"
    return mejor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/week9_det_cbam32.pth")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--casos", nargs="*", default=None, help="casos fijos en vez del split")
    ap.add_argument("--min-dist", type=float, default=2.0, help="mm minimos al principal")
    ap.add_argument("--top", type=int, default=2, help="fragmentos mas lejanos por caso")
    ap.add_argument("--confidence", type=float, default=0.20)
    ap.add_argument("--cobertura", type=float, default=0.5,
                    help="fraccion del fragmento que debe quedar dentro de la caja")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--output-dir", default="outputs/figures/week9_desplazados")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = detection_model_from_checkpoint(ck).eval()
    casos = args.casos or [str(c).zfill(3) for c in json.load(open(args.splits))[args.split]]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    region_de = {name: (i, lo, hi, main) for i, (name, (lo, hi, main)) in enumerate(REGIONS.items())}

    filas = []
    for cid in casos:
        caso = load_case(cid)
        lab = caso.label
        distancias = [d for d in measure_fragment_distances(lab, caso.spacing_xyz)
                      if d.distance_mm >= args.min_dist]
        distancias = sorted(distancias, key=lambda d: -d.distance_mm)[:args.top]
        if not distancias:
            continue
        con_hueso = np.flatnonzero((lab != 0).any(axis=(1, 2)))
        for d in distancias:
            clase, lo, hi, _ = region_de[d.region]
            area = (lab == d.fragment_label).sum(axis=(1, 2))
            z = int(area.argmax())
            posicion = (z - con_hueso[0]) / max(con_hueso[-1] - con_hueso[0], 1)
            img = resize_slice(window_hu(caso.image[z]), args.size).astype(np.float32)
            lab2 = resize_label_slice(lab[z], args.size)
            frag = lab2 == d.fragment_label
            if not frag.any():          # fragmento que desaparece al reducir a 256
                continue
            x = torch.from_numpy(img)[None, None]
            with torch.no_grad():
                salida = model(x)
            todas = decode_predictions(salida, confidence=args.confidence)[0]
            una = decode_predictions(salida, confidence=args.confidence, max_por_clase=1)[0]
            v_todas = veredicto(frag, todas, clase, args.cobertura)
            v_una = veredicto(frag, una, clase, args.cobertura)
            dentro = RANGO_ENTRENAMIENTO[0] <= posicion <= RANGO_ENTRENAMIENTO[1]
            filas.append({"caso": cid, "z": z, "region": d.region, "fragmento": d.fragment_label,
                          "dist_mm": round(d.distance_mm, 1), "pixeles_256": int(frag.sum()),
                          "posicion_axial": round(float(posicion), 2),
                          "dentro_rango_entrenamiento": dentro,
                          "veredicto_todas": v_todas["veredicto"],
                          "cobertura_todas": round(v_todas["cobertura"], 2),
                          "caja_todas": v_todas["clase_caja"],
                          "veredicto_1_por_clase": v_una["veredicto"],
                          "cobertura_1_por_clase": round(v_una["cobertura"], 2)})

            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            region_mask = (lab2 >= lo) & (lab2 <= hi)
            ax.contour(region_mask & ~frag, levels=[0.5], colors=COLORES[clase], linewidths=0.8)
            ax.contour(frag, levels=[0.5], colors="yellow", linewidths=1.6)
            for c in todas:
                x0, y0, x1, y1 = [v * args.size for v in c["box"]]
                k = int(c["class_id"])
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1.4,
                                       edgecolor=COLORES[k]))
                ax.text(x0, max(y0 - 2, 0), f"{REGION_NAMES[k]} {c['score']:.2f}", fontsize=7,
                        color="white", bbox={"facecolor": COLORES[k], "alpha": .75, "pad": 1})
            ax.set_title(f"caso {cid}, corte {z} — fragmento {d.fragment_label} ({d.region}) a "
                         f"{d.distance_mm:.1f} mm\nveredicto: {v_todas['veredicto']} (todas) / "
                         f"{v_una['veredicto']} (1 por clase)"
                         + ("" if dentro else " · corte fuera del rango de entrenamiento"),
                         fontsize=9)
            ax.axis("off"); fig.tight_layout()
            fig.savefig(out / f"desplazado_{cid}_{d.fragment_label}.png", dpi=100); plt.close(fig)
            print(f"caso {cid} z={z} frag {d.fragment_label} {d.region} {d.distance_mm:.1f} mm "
                  f"(posición {posicion:.2f}) → {v_todas['veredicto']} / {v_una['veredicto']}")

    df = pd.DataFrame(filas)
    df.to_csv(out / "resumen.csv", index=False)
    if df.empty:
        print("No hay fragmentos desplazados que cumplan el criterio."); return
    print(f"\n{len(df)} fragmentos a ≥ {args.min_dist} mm de su principal ({args.split})")
    for col, nombre in (("veredicto_todas", "todas las cajas"), ("veredicto_1_por_clase", "1 por clase")):
        print(f"  {nombre:16s}: {df[col].value_counts().to_dict()}")
    print("  dentro del rango axial de entrenamiento:",
          df["dentro_rango_entrenamiento"].value_counts().to_dict())
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
