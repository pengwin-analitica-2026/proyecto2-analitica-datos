"""Comprueba el detector geométrico de orientación contra las cabeceras de los 100 casos.

``orientacion.estimate_angle`` no usa etiquetas ni red. La cabecera de cada .mha dice si el
volumen está guardado girado 180 grados (lista ``LPI_CASES``), así que hay verdad de terreno
gratis para el sentido: los RAI deben dar cerca de 0 y los LPI cerca de +-180. Los que queden
lejos de ambos son candidatos a paciente girado.

Uso:
    python src/validar_orientacion.py --cache-dir data/cache/s10/cache_384 --output outputs/s10/orientacion_100.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orientacion import estimate_angle, is_rotated  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    with open(cache_dir.parent / "index.csv", newline="") as stream:
        index = list(csv.DictReader(stream))
    rows = []
    for row in index:
        case_id = row["case_id"]
        with np.load(cache_dir / f"{case_id}.npz") as data:
            images = data["images"]
        size = images.shape[-1]
        mm_y = float(row["shape_y"]) * float(row["spacing_y_mm"]) / size
        mm_x = float(row["shape_x"]) * float(row["spacing_x_mm"]) / size
        angle, detail = estimate_angle(images, mm_y, mm_x)
        expected = 180.0 if is_rotated(case_id) else 0.0
        error = abs(((angle - expected + 180) % 360) - 180)          # distancia al valor de la cabecera
        rows.append({"caso": case_id, "split": row["split"], "cabecera": "LPI" if expected else "RAI",
                     "angulo": round(angle, 1), "desvio_respecto_a_cabecera": round(error, 1),
                     "separacion_mm": round(detail.get("separacion_mm", float("nan")), 1),
                     "alargamiento": round(detail.get("alargamiento", float("nan")), 2)})
        print(rows[-1], flush=True)
    errors = np.asarray([r["desvio_respecto_a_cabecera"] for r in rows])
    summary = {"n": len(rows), "sentido_correcto (desvio < 90)": int((errors < 90).sum()),
               "desvio_mediana": float(np.median(errors)), "desvio_p90": float(np.percentile(errors, 90)),
               "desvio_max": float(errors.max()),
               "casos_con_desvio_mayor_a_15": [(r["caso"], r["split"], r["angulo"], r["desvio_respecto_a_cabecera"])
                                               for r in rows if r["desvio_respecto_a_cabecera"] > 15]}
    print(json.dumps(summary, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps({"resumen": summary, "casos": rows}, indent=1))


if __name__ == "__main__":
    main()
