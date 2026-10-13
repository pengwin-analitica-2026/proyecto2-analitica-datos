"""Prueba corta en Kaggle de datos, transferencia, CUDA y estabilidad numérica."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

import torch


CODIGO = Path("/kaggle/temp/codigo")
ENTRADA = Path("/kaggle/input")
SALIDA = Path("/kaggle/working")


def main() -> None:
    os.chdir(CODIGO)
    subprocess.run(["nvidia-smi"], check=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")
    indices = [Path(p) for p in glob.glob(str(ENTRADA / "**" / "index.csv"), recursive=True)]
    checkpoints = list(ENTRADA.rglob("week9_25d_rotacion_fisica_desde_cero_w32_e35_s42.pth"))
    if len(indices) != 1 or len(checkpoints) != 1:
        raise RuntimeError(f"Datos ambiguos: indices={indices}, checkpoints={checkpoints}")
    cache_root = indices[0].parent
    if len(list((cache_root / "cache_384").glob("*.npz"))) != 100:
        raise RuntimeError("El caché privado no contiene 100 casos.")
    comando = [
        sys.executable, "src/train_s10.py",
        "--cache-root", str(cache_root), "--size", "384", "--epochs", "2",
        "--lr", "0.002", "--base-channels", "32", "--cbam", "1",
        "--secondary-weight", "3", "--augment", "--cosine", "--batch-size", "2",
        "--per-epoch", "250", "--eval-every", "1", "--context", "1", "--seed", "42",
        "--aug-geo", "1.0", "--aug-rot-deg", "35", "--canonical", "--aug-mm",
        "--ema", "0", "--early-stopping-patience", "0",
        "--checkpoint-metric", "instance_dice",
        "--week9-checkpoint", str(checkpoints[0]), "--freeze-week9-epochs", "1",
        "--output", str(SALIDA / "smoke_best.pth"),
        "--last-output", str(SALIDA / "smoke_epoch2.pth"),
        "--metrics-output", str(SALIDA / "smoke_historial.json"),
    ]
    resultado = subprocess.run(comando, cwd=CODIGO)
    if resultado.returncode != 0:
        raise RuntimeError(f"La prueba corta falló con código {resultado.returncode}.")
    print("PRUEBA CORTA GPU COMPLETADA SIN NaN", flush=True)


if __name__ == "__main__":
    main()
