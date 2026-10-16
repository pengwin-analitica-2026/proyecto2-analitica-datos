"""Latencia por imagen del pipeline (modelo + NMS propio), en CPU y en GPU (Semana 11).

El enunciado pide medir y reportar la latencia por imagen en los dos escenarios. Se mide un corte a
la vez (batch 1), con pasadas de calentamiento descartadas, y se reporta la mediana y el percentil 90
de ``--repeats`` pasadas. En GPU se sincroniza antes de leer el reloj.

Uso:

    python src/measure_latency.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \\
        --volume outputs/demo/h8_aumm_noise01_noema_w32_e60_s42/024/volume.npz --device cpu --device cuda

Escribe (y acumula) las filas en ``outputs/demo/latency.json``, que el dashboard muestra. Si pides
``cuda`` y no hay GPU, lo dice y lo omite; no inventa la cifra.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_volume import load_pipeline, predict_volume


def sample_slices(volume_path: str | None, size: int, count: int, seed: int = 0) -> np.ndarray:
    """Cortes reales del volumen exportado (con hueso) o, sin volumen, ruido del mismo tamaño."""
    if volume_path:
        with np.load(volume_path) as data:
            images, semantic = data["images"], data["semantic"]
        with_bone = np.flatnonzero((semantic > 0).any(axis=(1, 2)))
        rng = np.random.default_rng(seed)
        return images[rng.choice(with_bone, size=min(count, len(with_bone)), replace=False)]
    return np.random.default_rng(seed).integers(0, 255, size=(count, size, size)).astype(np.uint8)


def cpu_name() -> str:
    """Nombre legible de la CPU (Mac, Linux o, en último caso, la arquitectura)."""
    import subprocess
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(checkpoint: str, device_name: str, tta: bool, slices: np.ndarray, repeats: int,
            warmup: int = 3) -> dict:
    device = torch.device(device_name)
    model, device, info = load_pipeline(checkpoint, device, tta=tta)
    times = []
    for i in range(warmup + repeats):
        image = slices[i % len(slices)]
        sync(device)
        started = time.perf_counter()
        predict_volume(model, image[None], device, batch_size=1)
        sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - started) * 1000.0)
    label = cpu_name()
    if device.type == "cuda":
        label = torch.cuda.get_device_name(0)
    return {"Modelo": info["checkpoint"], "Dispositivo": device.type.upper(), "Hardware": label,
            "TTA": "sí" if tta else "no", "Hilos CPU": torch.get_num_threads() if device.type == "cpu" else "—",
            "Cortes por segundo": round(1000.0 / float(np.median(times)), 2),
            "Mediana (ms)": round(float(np.median(times)), 1),
            "p90 (ms)": round(float(np.percentile(times, 90)), 1),
            "Pasadas": repeats, "Tamaño": f"{slices.shape[-1]}×{slices.shape[-1]}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--volume", default=None, help="volume.npz de una exportación (cortes reales).")
    parser.add_argument("--device", action="append", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--output", default="outputs/demo/latency.json")
    args = parser.parse_args()

    slices = sample_slices(args.volume, args.size, args.repeats)
    rows = []
    for device_name in args.device or ["cpu", "cuda"]:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("cuda: no hay GPU en esta máquina; se omite (no se estima)")
            continue
        for tta in (False, True):
            row = measure(args.checkpoint, device_name, tta, slices, args.repeats)
            rows.append(row)
            print(f"{row['Dispositivo']} TTA {row['TTA']}: mediana {row['Mediana (ms)']} ms, p90 {row['p90 (ms)']} ms")

    path = Path(args.output)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    keys = {(r["Modelo"], r["Dispositivo"], r["Hardware"], r["TTA"]) for r in rows}
    kept = [r for r in previous if (r["Modelo"], r["Dispositivo"], r["Hardware"], r["TTA"]) not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kept + rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {path}")


if __name__ == "__main__":
    main()
