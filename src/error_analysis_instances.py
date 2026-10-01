"""Análisis de error de la segmentación por fragmento (Semana 10).

Para cada fragmento del ground truth en validación calcula el Dice emparejado y
lo clasifica según qué falló: no detectado, fusionado con otro fragmento,
partido en varios, o solo bordes imprecisos. Agrupa por tamaño y región, y
dibuja los cortes con peor Dice medio (imagen | GT | predicción).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instance_postprocess import primary_secondary_to_instances, semantic_to_instances
from week10_multitask import INSTANCE_MODES, PengwinMultiTaskModel

BUCKETS = [(0, 50, "<50 px (muy pequeño)"), (50, 200, "50–200 px"),
           (200, 800, "200–800 px"), (800, 10 ** 9, ">800 px (grande)")]
REGION = {0: "sacro", 1: "coxal izq", 2: "coxal der"}


def region_of(value: int) -> int:
    return min((value - 1) // 10, 2)


def analyse_slice(pred: np.ndarray, truth: np.ndarray) -> list[dict]:
    rows = []
    for low, high in ((1, 10), (11, 20), (21, 30)):
        gt_ids = [int(v) for v in np.unique(truth) if low <= v <= high]
        pred_ids = [int(v) for v in np.unique(pred) if low <= v <= high]
        used: set[int] = set()
        for gt_id in gt_ids:
            gt = truth == gt_id
            best, best_iou = None, 0.0
            for pid in pred_ids:
                if pid in used:
                    continue
                cand = pred == pid
                iou = np.logical_and(gt, cand).sum() / max(np.logical_or(gt, cand).sum(), 1)
                if iou > best_iou:
                    best, best_iou = pid, iou
            area = int(gt.sum())
            if best is None:
                rows.append({"gt_id": gt_id, "area": area, "dice": 0.0, "fallo": "no detectado"})
                continue
            used.add(best)
            cand = pred == best
            dice = 2 * np.logical_and(gt, cand).sum() / max(gt.sum() + cand.sum(), 1)
            # ¿La instancia predicha cubre además otro fragmento GT de la región?
            other_gt = np.isin(truth, [g for g in gt_ids if g != gt_id])
            fused = np.logical_and(cand, other_gt).sum() > 0.2 * other_gt.sum() if other_gt.any() else False
            # ¿El fragmento GT quedó repartido en varias instancias predichas?
            covering = [pid for pid in pred_ids if np.logical_and(pred == pid, gt).sum() > 0.2 * area]
            split = len(covering) > 1
            fallo = "fusionado" if fused else "partido" if split else "bordes"
            rows.append({"gt_id": gt_id, "area": area, "dice": float(dice), "fallo": fallo})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache/week10_multitask_slices_x12.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", default="outputs/error_analysis")
    parser.add_argument("--worst", type=int, default=5)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    mode = ckpt.get("instance_mode", "ids")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PengwinMultiTaskModel(base_channels=int(ckpt["base_channels"]), instance_mode=mode)
    model.load_state_dict(ckpt["model"]); model.to(device).eval()

    data = np.load(args.cache)
    mask = data["split_names"] == args.split
    images = data["images"][mask].astype(np.float32)
    labels = data["labels"][mask].astype(np.int64)
    case_ids, z_indices = data["case_ids"][mask], data["z_indices"][mask]
    preds = []
    with torch.inference_mode():
        for s in range(0, len(images), 16):
            out = model(torch.from_numpy(images[s:s + 16]).to(device))
            sem = out["semantic"].argmax(1).cpu().numpy()
            inst = out["instance"].argmax(1).cpu().numpy()
            preds.append(primary_secondary_to_instances(inst, sem) if mode == "primary"
                         else semantic_to_instances(sem))
    preds = np.concatenate(preds)

    rows, per_slice = [], []
    for i, (p, t) in enumerate(zip(preds, labels)):
        r = analyse_slice(p, t)
        for row in r:
            row.update({"slice": i, "case": str(case_ids[i]), "z": int(z_indices[i]),
                        "region": REGION[region_of(row["gt_id"])],
                        "tipo": "primario" if row["gt_id"] in (1, 11, 21) else "secundario"})
        rows.extend(r)
        if r:
            per_slice.append((float(np.mean([x["dice"] for x in r])), i))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "fragmentos_val.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    dices = np.array([r["dice"] for r in rows]); areas = np.array([r["area"] for r in rows])
    print(f"Fragmentos GT en {args.split}: {len(rows)} | Dice medio: {dices.mean():.4f}\n")
    print("Por tamaño del fragmento (px a 256×256):")
    for lo, hi, name in BUCKETS:
        sel = (areas >= lo) & (areas < hi)
        if sel.any():
            perdida = ((1 - dices[sel]).sum() / (1 - dices).sum()) * 100
            print(f"  {name:22s} n={sel.sum():4d}  dice={dices[sel].mean():.3f}  "
                  f"aporta {perdida:5.1f}% de la pérdida total")
    print("\nPor tipo de fallo:")
    for fallo in ("bordes", "fusionado", "partido", "no detectado"):
        sel = np.array([r["fallo"] == fallo for r in rows])
        if sel.any():
            perdida = ((1 - dices[sel]).sum() / (1 - dices).sum()) * 100
            print(f"  {fallo:14s} n={sel.sum():4d}  dice={dices[sel].mean():.3f}  "
                  f"aporta {perdida:5.1f}% de la pérdida total")
    print("\nPor tipo y región:")
    for tipo in ("primario", "secundario"):
        for reg in REGION.values():
            sel = np.array([r["tipo"] == tipo and r["region"] == reg for r in rows])
            if sel.any():
                print(f"  {tipo:10s} {reg:10s} n={sel.sum():4d}  dice={dices[sel].mean():.3f}")

    per_slice.sort()
    worst = per_slice[:args.worst]
    fig, axes = plt.subplots(len(worst), 3, figsize=(10, 3.3 * len(worst)))
    cmap = plt.get_cmap("tab20", 31)
    for row_axes, (d, i) in zip(np.atleast_2d(axes), worst):
        row_axes[0].imshow(images[i, 0], cmap="gray"); row_axes[0].set_title(f"caso {case_ids[i]} z={z_indices[i]}  dice={d:.2f}")
        row_axes[1].imshow(np.ma.masked_equal(labels[i], 0), cmap=cmap, vmin=0, vmax=30); row_axes[1].set_title("GT")
        row_axes[2].imshow(np.ma.masked_equal(preds[i], 0), cmap=cmap, vmin=0, vmax=30); row_axes[2].set_title("predicción")
        for ax in row_axes: ax.axis("off")
    plt.tight_layout(); fig.savefig(out / "peores_cortes.png", dpi=120)
    print(f"\nPeores {len(worst)} cortes: {[str(case_ids[i]) + '/z' + str(z_indices[i]) for _, i in worst]}")
    print(f"Guardado: {out/'fragmentos_val.csv'} y {out/'peores_cortes.png'}")


if __name__ == "__main__":
    main()
