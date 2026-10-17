"""Arma los kernels de Kaggle que evalúan la cuantización de H8.

Datos: datasets privados y planos (``<caso>.npz``, ``index.csv``, ``splits.json`` y el checkpoint de H8):

- ``<usuario>/pengwin-cuantizacion-h8``: los 15 casos de validación y los 8 de entrenamiento con que se
  calibra int8 (``src/build_case_cache.py --sizes 384``);
- ``<usuario>/pengwin-cuantizacion-h8-test``: los 15 casos de test y los mismos 8 de entrenamiento.
  Solo para reportar una vez la variante elegida en validación.

int8 de PyTorch solo corre en CPU (y la demo corre en CPU en Lambda): sus kernels van sin GPU. fp16 y
bf16 se miden en GPU, que es donde rinden; fp32 va en los dos como referencia.

    python kaggle/h8/armar_cuantizacion.py --salida build/kaggle_cuant --usuario gustaveloper [--kernels gpu-val]
    kaggle kernels push -p build/kaggle_cuant/<kernel>
    kaggle kernels output <usuario>/pengwin-cuant-h8-<kernel> -p outputs/h8/cuantizacion/<kernel>

Ver ``docs/cuantizacion.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from armar import AQUI, empaquetar_src

VAL, TEST = "pengwin-cuantizacion-h8", "pengwin-cuantizacion-h8-test"
# nombre: (variantes, dispositivo, split, dataset)
KERNELS = {
    "fp32": ("fp32", "cpu", "val", VAL),
    "int8": ("int8", "cpu", "val", VAL),
    "int8-todo": ("int8_todo", "cpu", "val", VAL),
    "bf16": ("bf16", "cpu", "val", VAL),
    "fp16": ("fp16", "cpu", "val", VAL),
    "gpu-val": ("fp32 fp16 bf16", "cuda", "val", VAL),
    "test-int8": ("int8", "cpu", "test", TEST),
    "test-fp32-gpu": ("fp32", "cuda", "test", TEST),
    "test-fp16-gpu": ("fp16", "cuda", "test", TEST),   # la elegida en validación para desplegar en GPU
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--salida", required=True)
    parser.add_argument("--usuario", required=True)
    parser.add_argument("--kernels", nargs="*", default=list(KERNELS))
    args = parser.parse_args()
    payload = empaquetar_src()
    for nombre in args.kernels:
        variantes, dispositivo, split, dataset = KERNELS[nombre]
        codigo = ((AQUI / "plantilla_cuantizar.py").read_text(encoding="utf8")
                  .replace("__PAYLOAD__", payload).replace("__VARIANTES__", variantes)
                  .replace("__DISPOSITIVO__", dispositivo).replace("__SPLIT__", split))
        carpeta = Path(args.salida) / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "script.py").write_text(codigo, encoding="utf8", newline="\n")
        slug = f"pengwin-cuant-h8-{nombre}"
        metadatos = {"id": f"{args.usuario}/{slug}", "title": slug, "code_file": "script.py",
                     "language": "python", "kernel_type": "script", "is_private": "true",
                     "enable_gpu": "true" if dispositivo == "cuda" else "false", "enable_internet": "true",
                     "dataset_sources": [f"{args.usuario}/{dataset}"],
                     "competition_sources": [], "kernel_sources": []}
        (carpeta / "kernel-metadata.json").write_text(json.dumps(metadatos, indent=1))
        print(f"{nombre}: {variantes} · {dispositivo} · {split} → {slug}")


if __name__ == "__main__":
    main()
