"""Hoja de contacto para revisar a ojo los casos que el detector geométrico marca como inclinados.

Por cada caso dibuja, a escala física (mm), la proyección axial del hueso (máximo a lo largo
del volumen) y el corte con más hueso, con la dirección "posterior" que estima
``orientacion.estimate_angle`` (flecha) y la que dice la cabecera (línea punteada). No usa
etiquetas ni red. Por regla (bitácora 12.3) no se abren casos de test.

Uso:
    python src/contacto_orientacion.py --cache-dir data/cache/s10/cache_384 \
        --orientacion outputs/s10/orientacion_100.json --output outputs/s10/contacto_orientacion.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orientacion import estimate_angle, is_rotated  # noqa: E402


def axis_deviation(error_degrees: float) -> float:
    """Desvío del eje sin contar el sentido (0-90)."""
    value = abs(error_degrees) % 180
    return min(value, 180 - value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--orientacion", required=True, help="JSON de validar_orientacion.py")
    parser.add_argument("--min-eje", type=float, default=15.0, help="desvío mínimo del eje para entrar")
    parser.add_argument("--controles", type=int, default=3, help="casos sin desvío que se añaden de referencia")
    parser.add_argument("--bone-threshold", type=int, default=70)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache_dir = Path(args.cache_dir)
    with open(cache_dir.parent / "index.csv", newline="") as stream:
        index = {row["case_id"]: row for row in csv.DictReader(stream)}
    rows = [r for r in json.loads(Path(args.orientacion).read_text())["casos"] if r["split"] != "test"]
    for r in rows:
        r["eje"] = axis_deviation(r["desvio_respecto_a_cabecera"])
    flagged = sorted((r for r in rows if r["eje"] > args.min_eje), key=lambda r: -r["eje"])
    controls = sorted(rows, key=lambda r: r["eje"])[: args.controles]
    picks = flagged + controls

    fig, axes = plt.subplots(len(picks), 2, figsize=(7.4, 3.5 * len(picks)), squeeze=False)
    for (ax_proj, ax_cut), r in zip(axes, picks):
        case_id, row = r["caso"], index[r["caso"]]
        with np.load(cache_dir / f"{case_id}.npz") as data:
            images = data["images"]
        size = images.shape[-1]
        mm_y = float(row["shape_y"]) * float(row["spacing_y_mm"]) / size
        mm_x = float(row["shape_x"]) * float(row["spacing_x_mm"]) / size
        angle, _ = estimate_angle(images, mm_y, mm_x, bone_threshold=args.bone_threshold)
        header = 180.0 if is_rotated(case_id) else 0.0
        extent = (0, size * mm_x, size * mm_y, 0)
        bone = images >= args.bone_threshold
        z_best = int(np.argmax(bone.reshape(len(bone), -1).sum(axis=1)))
        cx, cy, length = size * mm_x / 2, size * mm_y / 2, 0.3 * min(size * mm_x, size * mm_y)
        for ax, picture, name in ((ax_proj, images.max(axis=0), "proyección axial"),
                                  (ax_cut, images[z_best], f"corte {z_best}")):
            ax.imshow(picture, cmap="gray", vmin=0, vmax=255, extent=extent, aspect="equal")
            ax.plot([cx, cx + length * math.sin(math.radians(header))],
                    [cy, cy + length * math.cos(math.radians(header))], "c--", lw=1.4)
            if np.isfinite(angle):
                ax.annotate("", xy=(cx + length * math.sin(math.radians(angle)), cy + length * math.cos(math.radians(angle))),
                            xytext=(cx, cy), arrowprops={"arrowstyle": "->", "color": "yellow", "lw": 2})
            if ax is ax_cut:
                ax.set_title(name, fontsize=9)
            ax.axis("off")
        kind = "CONTROL" if r in controls else "marcado"
        ax_proj.set_title(f"caso {case_id} ({r['split']}, {r['cabecera']}) {kind} | estimado {angle:.0f}°, "
                          f"eje desviado {r['eje']:.0f}°", fontsize=9, loc="left")
    fig.suptitle("Amarillo: posterior según el detector. Cian punteado: posterior según la cabecera.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(args.output, dpi=70)
    print("guardado", args.output, "casos", [r["caso"] for r in picks])


if __name__ == "__main__":
    main()
