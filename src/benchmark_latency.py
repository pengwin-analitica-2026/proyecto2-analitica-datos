"""Latencia por corte del pipeline (Semana 10, rendimiento de la demo).

Mide, con cortes reales del caché, el tiempo de una pasada de ``infer_slice``
(modelo + decodificación de detección + instancias por componentes) con lote 1,
sin y con TTA espejo y con 1 o 3 modelos (ensamble). Reporta mediana y p95 en ms.

Uso:
    python src/benchmark_latency.py --checkpoint outputs/s10/d5_ema_w32_e60_s42.pth \
        --cache-dir data/cache/s10/cache_384 --device cpu --output outputs/s10/latencia_cpu.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_cache import from_uint8, load_case_cache  # noqa: E402
from evaluate_volumes import cases_of_split, load_model  # noqa: E402
from inference_week10 import infer_slice  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(model, slices: list[np.ndarray], device: torch.device, warmup: int) -> dict:
    times = []
    for i, image in enumerate(slices):
        tensor = torch.from_numpy(from_uint8(image))[None, None].to(device)
        sync(device)
        start = time.perf_counter()
        infer_slice(model, tensor)
        sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    arr = np.asarray(times)
    return {"n": int(arr.size), "mediana_ms": round(float(np.median(arr)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
            "media_ms": round(float(arr.mean()), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Uno o más .pth; el primero se usa solo y todos juntos como ensamble")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-slices", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0, help="hilos de CPU (0 = por defecto)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    cache_dir = Path(args.cache_dir)
    case = load_case_cache(cache_dir, cases_of_split(cache_dir, "val")[0])
    images = case["images"]
    idx = np.linspace(len(images) * 0.25, len(images) * 0.75, args.n_slices + args.warmup).astype(int)
    slices = [images[i] for i in idx]

    result = {"device": str(device), "threads": torch.get_num_threads(),
              "tamano": list(slices[0].shape), "checkpoints": args.checkpoint, "config": {}}
    configs = [("1 modelo", args.checkpoint[:1], False), ("1 modelo + TTA", args.checkpoint[:1], True)]
    if len(args.checkpoint) > 1:
        configs.append((f"ensamble {len(args.checkpoint)} + TTA", args.checkpoint, True))
    for name, ckpts, tta in configs:
        model, _ = load_model(ckpts, device, tta)
        result["config"][name] = bench(model, slices, device, args.warmup)
        print(name, result["config"][name], flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
