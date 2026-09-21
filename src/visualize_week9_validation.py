"""Visualiza una predicción inicial en un corte de validación no visto."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week9_detection import DetectionModel, REGION_NAMES, boxes_from_label, decode_predictions

COLORS = ("#00BFFF", "#00C853", "#FF5252")


def draw(ax, item: dict, text: str, linestyle: str) -> None:
    box, cls = item["box"], int(item["class_id"])
    height, width = ax.images[0].get_array().shape
    xmin, ymin, xmax, ymax = box
    ax.add_patch(Rectangle((xmin * width, ymin * height), (xmax - xmin) * width,
                           (ymax - ymin) * height, fill=False, linewidth=2,
                           edgecolor=COLORS[cls], linestyle=linestyle))
    ax.text(xmin * width, max(0, ymin * height - 2), text, color="white", fontsize=8,
            bbox={"facecolor": COLORS[cls], "alpha": .75, "pad": 1})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cache/week9_detection_slices.npz")
    parser.add_argument("--checkpoint", default="outputs/week9_detection_initial.pth")
    parser.add_argument("--case-id", default=None, help="caso de validación; omitir para el primero")
    parser.add_argument("--all-predictions", action="store_true",
                        help="muestra todas las propuestas posteriores a NMS")
    parser.add_argument("--output", default="outputs/figures/week9_detection_validation.png")
    args = parser.parse_args()
    data = np.load(args.cache)
    ids, splits = data["case_ids"], data["split_names"]
    candidates = np.flatnonzero(splits == "val")
    if args.case_id:
        candidates = candidates[ids[candidates] == str(args.case_id).zfill(3)]
    if len(candidates) == 0:
        raise ValueError("No hay corte de validación para el case-id solicitado")
    index = int(candidates[len(candidates) // 2])
    image = torch.from_numpy(data["images"][index:index + 1].astype(np.float32))
    label = torch.from_numpy(data["labels"][index:index + 1].astype(np.int64))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DetectionModel(base_channels=checkpoint["base_channels"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        predicted = decode_predictions(model(image), confidence=.20)[0]
    if not args.all_predictions:
        # El visualizador de avance muestra una sola hipótesis legible por
        # región. La evaluación conserva todas las propuestas con NMS.
        predicted = [max((p for p in predicted if p["class_id"] == cls),
                         key=lambda p: p["score"])
                     for cls in range(3) if any(p["class_id"] == cls for p in predicted)]
    truth = boxes_from_label(label)[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image[0, 0], cmap="gray", vmin=0, vmax=1)
    for item in truth:
        cls = int(item["class_id"])
        draw(ax, item, f"GT {REGION_NAMES[cls]}", "solid")
    for item in predicted:
        cls = int(item["class_id"])
        draw(ax, item, f"Pred {REGION_NAMES[cls]} {item['score']:.2f}", "dashed")
    ax.set_title(f"Validación — caso {ids[index]}, corte {int(data['z_indices'][index])}\n"
                 "GT sólida; predicción discontinua")
    ax.axis("off")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
