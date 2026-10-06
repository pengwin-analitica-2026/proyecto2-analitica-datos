"""Análisis de error para el informe: matrices de confusión y los cinco peores casos.

Trabaja sobre lo que deja ``evaluate_volumes.py`` con ``--save-records`` y ``--save-predictions``;
no vuelve a inferir. Produce:

- matriz de confusión de la cabeza de clasificación (presencia de cada región por corte);
- matriz de confusión de la segmentación semántica (a qué región se asigna cada píxel);
- tabla de los cinco peores casos por Dice de fragmento, con qué les pasa;
- un panel CT / real / predicción por cada uno de esos casos.

Uso (misma configuración congelada de la evaluación; no ajusta nada):
    python src/evaluate_volumes.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
        --cache-dir data/cache/s10/cache_384 --split test --tta \
        --save-records /tmp/registros.pkl --save-predictions /tmp/pred
    python src/informe_test.py --records /tmp/registros.pkl --predictions /tmp/pred \
        --cache-dir data/cache/s10/cache_384 --canonical --output-dir outputs/s10/informe_h6_test
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_cache import load_case_cache  # noqa: E402
from evaluate_volumes import PRIMARY_IDS, SliceRecord, semantic_from_labels  # noqa: E402,F401

REGIONES = ("sacro", "coxal izquierdo", "coxal derecho")
CLASES = ("fondo",) + REGIONES
COLOR_REGION = {1: (0.16, 0.47, 0.84), 2: (0.92, 0.41, 0.20), 3: (0.11, 0.69, 0.48)}


def confusion_clasificacion(records: list) -> dict:
    """Por región: cortes en que está y se predice, está y no se predice, etc. Umbral: logit > 0."""
    logits = np.stack([r.cls_logits for r in records])
    target = np.stack([r.cls_target for r in records]) > 0.5
    pred = logits > 0
    out = {}
    for index, name in enumerate(REGIONES):
        tp = int((pred[:, index] & target[:, index]).sum())
        fp = int((pred[:, index] & ~target[:, index]).sum())
        fn = int((~pred[:, index] & target[:, index]).sum())
        tn = int((~pred[:, index] & ~target[:, index]).sum())
        out[name] = {"presente_y_predicha": tp, "ausente_y_predicha": fp, "presente_y_no_predicha": fn,
                     "ausente_y_no_predicha": tn,
                     "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
                     "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4)}
    out["f1_macro"] = round(float(np.mean([out[name]["f1"] for name in REGIONES])), 4)
    return out


def colorear(ids: np.ndarray) -> np.ndarray:
    """RGBA: un tono por región; el fragmento principal pleno y los secundarios más claros."""
    rgba = np.zeros(ids.shape + (4,), dtype=np.float32)
    for value in np.unique(ids):
        if value == 0:
            continue
        region = (int(value) - 1) // 10 + 1
        base = np.asarray(COLOR_REGION[region])
        orden = (int(value) - 1) % 10            # 0 = principal
        tono = base if orden == 0 else base + (1 - base) * min(0.35 + 0.12 * orden, 0.8)
        rgba[ids == value] = (*tono, 0.62)
    return rgba


def panel(path: Path, images: np.ndarray, labels: np.ndarray, instance: np.ndarray, cortes: list[tuple[int, float]],
          titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(cortes), 3, figsize=(11.5, 3.9 * len(cortes)), facecolor="#fcfcfb", squeeze=False)
    for fila, (z, dice) in zip(axes, cortes):
        ys, xs = np.where((labels[z] > 0) | (instance[z] > 0))
        y0, y1 = max(ys.min() - 14, 0), min(ys.max() + 14, images.shape[1])
        x0, x1 = max(xs.min() - 14, 0), min(xs.max() + 14, images.shape[2])
        for ax, capa, nombre in zip(fila, (None, labels[z], instance[z]), ("CT", "Real", "Predicción")):
            ax.imshow(images[z, y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
            if capa is not None:
                ax.imshow(colorear(capa[y0:y1, x0:x1]), interpolation="nearest")
                nombre += f" · {len(np.unique(capa)) - 1} fragmentos"
            ax.set_title(f"{nombre}" if capa is not None else f"corte {z} · Dice del corte {dice:.2f}",
                         fontsize=10, color="#0b0b0b", loc="left")
            ax.axis("off")
    fig.suptitle(titulo + "\nAzul: sacro · naranja: coxal izquierdo · verde: coxal derecho · tono claro: fragmento secundario",
                 fontsize=10.5, color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=75, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--canonical", action="store_true", help="El modelo se evaluó en orientación canónica")
    parser.add_argument("--peores", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.records, "rb") as stream:
        records = pickle.load(stream)
    por_caso = defaultdict(list)
    for record in records:
        por_caso[record.case_id].append(record)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.zeros((4, 4), dtype=np.int64)
    casos = []
    detalle = {}
    for case_id, recs in sorted(por_caso.items()):
        case = load_case_cache(args.cache_dir, case_id, canonical=args.canonical)
        labels = case["labels"]
        with np.load(Path(args.predictions) / f"{case_id}.npz") as data:
            semantic, instance = data["semantic"], data["instance"]
        gt = semantic_from_labels(labels)
        conf = np.bincount(gt.ravel().astype(np.int64) * 4 + semantic.ravel().astype(np.int64), minlength=16).reshape(4, 4)
        confusion += conf
        frag = [f for r in recs for f in r.fragments]            # (id real, dice, iou, área)
        prim = [f[1] for f in frag if f[0] in PRIMARY_IDS]
        sec = [f[1] for f in frag if f[0] not in PRIMARY_IDS]
        hueso = max(conf[1:].sum(), 1)
        por_corte = [(r.z, float(np.mean([f[1] for f in r.fragments])), len(r.fragments), r.rel_pos)
                     for r in recs if r.fragments]
        casos.append({
            "caso": case_id, "cortes": len(recs), "fragmentos_por_corte": len(frag),
            "dice_fragmento": round(float(np.mean([f[1] for f in frag])), 4),
            "dice_primario": round(float(np.mean(prim)), 4),
            "dice_secundario": round(float(np.mean(sec)), 4) if sec else None,
            "fragmentos_secundarios_por_corte": len(sec),
            "secundarios_sin_pareja_%": round(100 * float(np.mean([d == 0 for d in sec])), 1) if sec else None,
            "hueso_asignado_a_otra_region_%": round(100 * float((conf[1:, 1:].sum() - np.trace(conf[1:, 1:])) / hueso), 2),
            "hueso_predicho_como_fondo_%": round(100 * float(conf[1:, 0].sum() / hueso), 2),
            "dice_centro_15_85": round(float(np.mean([f[1] for r in recs if 0.15 <= r.rel_pos <= 0.85 for f in r.fragments])), 4),
            "dice_extremos": round(float(np.mean([f[1] for r in recs if not 0.15 <= r.rel_pos <= 0.85 for f in r.fragments] or [np.nan])), 4),
        })
        detalle[case_id] = (case["images"], labels, instance, por_corte)

    casos.sort(key=lambda fila: fila["dice_fragmento"])
    peores = casos[: args.peores]
    for fila in peores:
        images, labels, instance, por_corte = detalle[fila["caso"]]
        centro = [p for p in por_corte if 0.15 <= p[3] <= 0.85] or por_corte
        ordenados = sorted(centro, key=lambda p: p[1])
        elegidos = [ordenados[len(ordenados) // 10], ordenados[len(ordenados) // 2]]      # uno malo y uno típico
        panel(out_dir / f"peor_{fila['caso']}.png", images, labels, instance, [(p[0], p[1]) for p in elegidos],
              f"Caso {fila['caso']} · Dice por fragmento {fila['dice_fragmento']:.3f} "
              f"(arriba: corte del décimo percentil; abajo: corte mediano)")
        fila["panel"] = f"peor_{fila['caso']}.png"

    todos = [f[1] for r in records for f in r.fragments]
    resultado = {
        "registros": args.records, "n_cortes": len(records), "n_casos": len(por_caso),
        "dice_fragmento_global": round(float(np.mean(todos)), 4),
        "clasificacion": confusion_clasificacion(records),
        "confusion_semantica": {"clases": list(CLASES), "filas_real_columnas_predicho": confusion.tolist()},
        "peores_casos": peores, "todos_los_casos": casos,
    }
    (out_dir / "informe.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(json.dumps({k: resultado[k] for k in ("n_cortes", "dice_fragmento_global", "clasificacion")}, indent=1, ensure_ascii=False))
    for fila in peores:
        print(fila)


if __name__ == "__main__":
    main()
