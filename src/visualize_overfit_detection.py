"""Guarda overlays de cajas ground truth y predichas para Semana 9.

Uso:
    python src/visualize_overfit_detection.py --case-id 001 \
        --checkpoint outputs/week9_overfit_detection.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_overfit_detection import make_batch
from week9_detection import DetectionModel, REGION_NAMES, boxes_from_label, decode_predictions


COLORS = ("#00BFFF", "#00C853", "#FF5252")


def draw_box(ax, box: list[float], class_id: int, label: str, style: str) -> None:
    height, width = ax.images[0].get_array().shape
    xmin, ymin, xmax, ymax = box
    ax.add_patch(Rectangle((xmin * width, ymin * height), (xmax - xmin) * width,
                           (ymax - ymin) * height, fill=False, linewidth=2,
                           linestyle=style, edgecolor=COLORS[class_id]))
    ax.text(xmin * width, max(0, ymin * height - 3), label, color="white", fontsize=8,
            bbox={"facecolor": COLORS[class_id], "alpha": 0.75, "pad": 1})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="001")
    parser.add_argument("--checkpoint", default="outputs/week9_overfit_detection.pth")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", default="outputs/figures/week9_detection_overfit.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DetectionModel(base_channels=checkpoint.get("base_channels", 32)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    x, labels = make_batch(args.case_id, size=checkpoint.get("input_size", 256),
                           max_slices=checkpoint.get("max_slices", 2))
    with torch.no_grad():
        predicted = decode_predictions(model(x.to(device)), confidence=args.confidence)
    truth = boxes_from_label(labels)

    n = len(x)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(15, 7), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= n:
            ax.axis("off")
            continue
        ax.imshow(x[index, 0], cmap="gray", vmin=0, vmax=1)
        for item in truth[index]:
            cls = int(item["class_id"])
            draw_box(ax, item["box"], cls, f"GT {REGION_NAMES[cls]}", "solid")
        for item in predicted[index]:
            cls = int(item["class_id"])
            draw_box(ax, item["box"], cls,
                     f"Pred {REGION_NAMES[cls]} {float(item['score']):.2f}", "dashed")
        ax.set_title(f"Corte del batch {index + 1}")
        ax.axis("off")
    fig.suptitle("Semana 9: cajas ground truth (sólida) y predicción (discontinua)")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
