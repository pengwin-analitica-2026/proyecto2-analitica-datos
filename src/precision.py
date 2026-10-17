"""Precisión numérica de H8 según el hardware: fp32, int8 híbrido (CPU) o fp16 (GPU).

Es el mismo modelo con los mismos pesos; solo cambia cómo se guardan y calculan los números. En
validación y en test las tres dan el mismo Dice (``docs/cuantizacion.md``); lo que cambia es la
velocidad, y cada una rinde en un hardware distinto:

- ``int8``: enteros de 8 bits, cuantización estática después del entrenamiento. Solo CPU (es lo
  único que PyTorch tiene para int8). Las 4 convoluciones transpuestas del decodificador quedan en
  fp32 porque en int8 PyTorch no tiene una implementación rápida y el modelo terminaba más lento que
  el original. 2.3 veces más rápido que fp32 en la CPU de Kaggle.
- ``fp16``: punto flotante de 16 bits. Para GPU (1.8 veces más rápido en una T4); en una CPU sin
  instrucciones de 16 bits es mucho más lenta que fp32.
- ``fp32``: el modelo tal como se entrenó, de referencia.
- ``auto``: fp16 si hay GPU, int8 si no.

El int8 se puede guardar ya cuantizado (``exportar_int8``, TorchScript, ~5 MB) para cargarlo sin
calibrar. El archivo depende de la versión de torch con que se hizo: si no coincide con la que lo
carga, o no existe, se vuelve a cuantizar en unos segundos con los 128 cortes de **entrenamiento** de
``outputs/s10/calibracion_int8_h8.npz`` (los mismos que se usaron en Kaggle).

    python src/precision.py exportar --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
CALIBRACION = ROOT / "outputs" / "s10" / "calibracion_int8_h8.npz"
PRECISIONES = ("auto", "fp32", "int8", "fp16")


def elegir(pedida: str, device: torch.device) -> str:
    """La precisión concreta: ``auto`` → fp16 en GPU, int8 en CPU."""
    pedida = (pedida or "auto").lower()
    if pedida not in PRECISIONES:
        raise ValueError(f"Precisión desconocida: {pedida} (opciones: {', '.join(PRECISIONES)})")
    if pedida == "auto":
        return "fp16" if device.type == "cuda" else "int8"
    if pedida == "int8" and device.type != "cpu":
        raise ValueError("int8 de PyTorch solo corre en CPU; en GPU usa fp16.")
    return pedida


def ruta_int8(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(f"{checkpoint.stem}_int8.pt")


def cortes_calibracion(ruta: str | Path = CALIBRACION) -> torch.Tensor:
    from build_case_cache import from_uint8
    with np.load(ruta) as datos:
        return torch.from_numpy(from_uint8(datos["cortes"]))[:, None]


class ConAtributos(nn.Module):
    """Envuelve un modelo convertido (FX o TorchScript) y le devuelve los atributos que usa el pipeline."""

    def __init__(self, modelo: nn.Module, base: nn.Module):
        super().__init__()
        self.modelo = modelo
        self.instance_mode = getattr(base, "instance_mode", "primary")
        self.in_channels = getattr(base, "in_channels", 1)
        self.canonical = bool(getattr(base, "canonical", False))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict(self.modelo(x))


def _cuantizar(base: nn.Module, calibracion: torch.Tensor) -> nn.Module:
    from cuantizar import quantize_int8
    return quantize_int8(base.cpu().eval(), calibracion, keep_float=False).eval()


def exportar_int8(checkpoint: str | Path, salida: str | Path | None = None,
                  calibracion: str | Path = CALIBRACION) -> Path:
    """Cuantiza H8 a int8 híbrido y lo guarda como TorchScript, listo para cargar sin calibrar."""
    from evaluate_volumes import load_model
    salida = Path(salida or ruta_int8(checkpoint))
    base, _ = load_model([str(checkpoint)], torch.device("cpu"), tta=False)
    cuantizado = _cuantizar(base, cortes_calibracion(calibracion))
    with torch.inference_mode():
        trazado = torch.jit.freeze(torch.jit.trace(cuantizado, cortes_calibracion(calibracion)[:2], strict=False))
    extra = {"meta.json": json.dumps({"torch": torch.__version__, "checkpoint": Path(checkpoint).name,
                                      "precision": "int8", "transpuestas": "fp32",
                                      "instance_mode": getattr(base, "instance_mode", "primary"),
                                      "in_channels": int(getattr(base, "in_channels", 1)),
                                      "canonical": bool(getattr(base, "canonical", False))})}
    torch.jit.save(trazado, str(salida), _extra_files=extra)
    return salida


def cargar_int8(checkpoint: str | Path, base: nn.Module) -> tuple[nn.Module, str]:
    """El int8 ya exportado si existe y es de esta versión de torch; si no, se cuantiza en el momento."""
    ruta = ruta_int8(checkpoint)
    if ruta.exists():
        extra = {"meta.json": ""}
        try:
            modelo = torch.jit.load(str(ruta), map_location="cpu", _extra_files=extra)
            meta = json.loads(extra["meta.json"] or "{}")
            if meta.get("torch") == torch.__version__ and meta.get("checkpoint") == Path(checkpoint).name:
                return ConAtributos(modelo.eval(), base), f"archivo {ruta.name}"
        except Exception:                          # archivo de otra versión de torch o dañado
            pass
    return ConAtributos(_cuantizar(base, cortes_calibracion()), base), "cuantizado al cargar"


def aplicar(base: nn.Module, precision: str, device: torch.device,
            checkpoint: str | Path | None = None) -> tuple[nn.Module, str]:
    """Devuelve (modelo en esa precisión, de dónde salió). ``base`` es el modelo fp32 sin TTA."""
    if precision == "fp32":
        return base, "checkpoint fp32"
    if precision == "fp16":
        from cuantizar import Precision
        modelo = Precision(base.to(device), torch.float16)
        modelo.canonical = bool(getattr(base, "canonical", False))
        return modelo, "fp16"
    if precision == "int8":
        if checkpoint is None:
            return ConAtributos(_cuantizar(base, cortes_calibracion()), base), "cuantizado al cargar"
        return cargar_int8(checkpoint, base)
    raise ValueError(precision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="accion", required=True)
    exp = sub.add_parser("exportar", help="Guarda H8 en int8 híbrido (TorchScript) junto al checkpoint.")
    exp.add_argument("--checkpoint", default=str(ROOT / "outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth"))
    exp.add_argument("--salida", default=None)
    args = parser.parse_args()
    inicio = time.perf_counter()
    salida = exportar_int8(args.checkpoint, args.salida)
    print(f"{salida} ({salida.stat().st_size / 1e6:.1f} MB, torch {torch.__version__}) "
          f"en {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
