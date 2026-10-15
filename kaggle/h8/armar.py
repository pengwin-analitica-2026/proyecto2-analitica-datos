"""Arma los kernels de Kaggle con los que se entrenó y evaluó H8.

Cada kernel es una de las plantillas de esta carpeta con ``src/`` del repo embebido (el repositorio es
privado, así que el código viaja dentro del script). Los datos salen de dos datasets privados:

- ``<usuario>/pengwin-cache-trainval``: caché a 384 px de los 70 casos de entrenamiento y los 15 de
  validación (``src/build_case_cache.py --sizes 384 --modes stretch``), con su ``index.csv``;
- ``<usuario>/pengwin-h8-test``: caché de los 15 casos de test y el checkpoint de H8.

Uso, desde la raíz del repo:

    python kaggle/h8/armar.py --salida build/kaggle_h8 --usuario jacobodm
    kaggle kernels push -p build/kaggle_h8/h8                 # entrena H8 y lo valida (5 pruebas)
    kaggle kernels push -p build/kaggle_h8/h8test             # test, HD95/ASSD y ablación de CBAM
    kaggle kernels push -p build/kaggle_h8/h8dist             # distancia de separación en mm
    kaggle kernels push -p build/kaggle_h8/h8sam              # SAM zero-shot contra H8 (solo comparación)

Kaggle permite dos sesiones de GPU a la vez. Los resultados se bajan con
``kaggle kernels output <usuario>/pengwin-exp-<nombre> -p <carpeta>``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]

# nombre: (plantilla, opciones de entrenamiento, checkpoint, datasets)
KERNELS = {
    "h8": ("plantilla_entrenar_validar.py", "--aug-mm --ema 0 --aug-noise 0.1",
           "h8_aumm_noise01_noema_w32_e60_s42.pth", ["pengwin-cache-trainval"]),
    "h7ema": ("plantilla_entrenar_validar.py", "--aug-mm --ema 0.999",
              "h7ema_auggeo_rot35_canon_aumm_ema_w32_e60_s42.pth", ["pengwin-cache-trainval"]),
    "h6noema": ("plantilla_entrenar_validar.py", "--ema 0",
                "h6noema_auggeo_rot35_canon_noema_w32_e60_s42.pth", ["pengwin-cache-trainval"]),
    "h8test": ("plantilla_test_cbam_superficie.py", None, None, ["pengwin-cache-trainval", "pengwin-h8-test"]),
    "h8dist": ("plantilla_distancias.py", None, None, ["pengwin-cache-trainval", "pengwin-h8-test"]),
    "h8sam": ("plantilla_sam.py", None, None, ["pengwin-cache-trainval", "pengwin-h8-test"]),
}


def empaquetar_src() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(ROOT / "src", arcname="src",
                filter=lambda i: None if "__pycache__" in i.name or "diag_semana8" in i.name else i)
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--salida", required=True)
    parser.add_argument("--usuario", required=True)
    parser.add_argument("--kernels", nargs="*", default=list(KERNELS))
    args = parser.parse_args()

    payload = empaquetar_src()
    for nombre in args.kernels:
        plantilla, extra, pth, datasets = KERNELS[nombre]
        codigo = (AQUI / plantilla).read_text(encoding="utf8").replace("__PAYLOAD__", payload)
        if extra is not None:
            codigo = (codigo.replace("__NOMBRE__", nombre).replace("__EXTRA__", extra)
                      .replace("__PTH__", pth))
        carpeta = Path(args.salida) / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "script.py").write_text(codigo, encoding="utf8", newline="\n")
        slug = f"pengwin-exp-{nombre.replace('h8test', 'h8-test').replace('h8dist', 'h8-dist').replace('h8sam', 'h8-sam')}"
        metadatos = {"id": f"{args.usuario}/{slug}", "title": slug, "code_file": "script.py",
                     "language": "python", "kernel_type": "script", "is_private": "true",
                     "enable_gpu": "true", "enable_internet": "true",
                     "dataset_sources": [f"{args.usuario}/{d}" for d in datasets],
                     "competition_sources": [], "kernel_sources": []}
        (carpeta / "kernel-metadata.json").write_text(json.dumps(metadatos, indent=1))
        print(f"{nombre}: {carpeta} ({slug})")


if __name__ == "__main__":
    main()
