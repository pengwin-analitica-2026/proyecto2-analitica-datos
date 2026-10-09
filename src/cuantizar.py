"""Cuantización de H6: ¿se puede compactar el modelo sin perder Dice? (sugerencia del profesor).

Compara la misma red guardada con distinta precisión numérica:

- ``fp32``: la referencia, como se entrenó (32 bits por peso, punto flotante);
- ``fp16`` y ``bf16``: punto flotante de 16 bits, la mitad de memoria. fp16 guarda más decimales
  y bf16 más rango; en GPU aceleran, en CPU depende del procesador;
- ``int8``: enteros de 8 bits con escala por canal (cuantización estática después del
  entrenamiento, FX de PyTorch, backend x86). Un cuarto del tamaño. Necesita calibrar los rangos
  de las activaciones con cortes reales: se usan cortes de **entrenamiento**, nunca de val/test;
  Las convoluciones transpuestas del decodificador quedan en fp32 (ver ``quantize_int8``);
- ``int8_cbam_fp32``: igual que int8 pero el CBAM y las capas finales de cada cabeza quedan en
  fp32, por si la atención (sigmoides y productos) es lo que más sufre al cuantizar;
- ``int8_todo``: todo en int8, incluidas las convoluciones transpuestas. Se deja para mostrar
  el fallo que obligó a excluirlas.

Para cada variante mide, sobre los casos completos del split (por defecto validación):
Dice por fragmento, Dice semántico, F1 de clasificación, mAP50, Dice 3D, error de distancia en mm,
tamaño en MB y latencia por corte en CPU (lote 1, mediana y p90). La regla para adoptarla es la del
protocolo: no puede bajar el Dice por fragmento más que el ruido entre semillas (0.003).

No se prueba int4: PyTorch no tiene kernels de convolución int4 en CPU y es una técnica pensada
para los pesos de modelos de lenguaje enormes; con 2.7 M de parámetros el tamaño ya no es el
problema. Ver ``docs/cuantizacion.md``.

Uso (desde la raíz, con el caché de la Semana 10):

    python src/cuantizar.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \\
        --cache-dir data/cache/s10/cache_384 --split val --tta \\
        --output outputs/s10/cuantizacion_h6_val.json
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_cache import from_uint8, load_case_cache
from evaluate_volumes import cases_of_split, evaluate_case, load_model, summarize
from infer_volume import aggregate, read_index
from volume_postprocess import (cache_spacing_xyz, fragment_distance_table, postprocess_3d,
                                summarize_volume)

VARIANTES = ("fp32", "fp16", "bf16", "int8", "int8_cbam_fp32", "int8_todo")
# Capas que producen las salidas de cada cabeza (logits): en int8_cbam_fp32 quedan en fp32.
CAPAS_SALIDA = ("classification_head.classifier.4", "detection_head.predictor.3",
                "segmentation_head.semantic", "segmentation_head.instance", "segmentation_head.boundary",
                "segmentation_head.centers", "segmentation_head.offsets")
METRICAS = ("instance_dice_matched", "instance_dice_primary", "instance_dice_secondary",
            "segmentation_dice_macro", "classification_f1_macro", "detection_map50")


class Precision(nn.Module):
    """Corre la red en otra precisión y devuelve las salidas en fp32 (el post-proceso no cambia)."""

    def __init__(self, model: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model.to(dtype)
        self.dtype = dtype
        self.instance_mode = model.instance_mode
        self.in_channels = getattr(model, "in_channels", 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v.float() for k, v in self.model(x.to(self.dtype)).items()}


def calibration_slices(cache_dir: Path, splits_path: Path, n_cases: int, per_case: int,
                       canonical: bool, seed: int = 0) -> torch.Tensor:
    """Cortes con hueso de casos de **entrenamiento** para fijar los rangos de int8."""
    train = [str(c).zfill(3) for c in json.loads(splits_path.read_text())["train"]]
    rng = np.random.default_rng(seed)
    picked = []
    for case_id in rng.choice(train, size=min(n_cases, len(train)), replace=False):
        if not (cache_dir / f"{case_id}.npz").exists():
            continue
        case = load_case_cache(cache_dir, case_id, canonical=canonical)
        bone = np.flatnonzero((case["labels"] != 0).any(axis=(1, 2)))
        if len(bone):
            picked.append(case["images"][rng.choice(bone, size=min(per_case, len(bone)), replace=False)])
    if not picked:
        raise SystemExit(f"No hay casos de entrenamiento en {cache_dir} para calibrar int8.")
    return torch.from_numpy(from_uint8(np.concatenate(picked)))[:, None]


def quantize_int8(model: nn.Module, calibration: torch.Tensor, keep_float: bool,
                  transposed_int8: bool = False) -> nn.Module:
    """Cuantización estática post-entrenamiento (FX, backend x86, pesos por canal).

    Las ``ConvTranspose2d`` del decodificador quedan en fp32 salvo ``transposed_int8=True``:
    cuantizadas, la segmentación de los coxales desaparece (Dice de 0.99 a 0.00 frente a fp32 en el
    caso 014), mientras que con ellas en fp32 todo lo demás puede ir en int8 sin perder nada. Se
    localizó dejando en fp32 un bloque a la vez; ``int8_todo`` deja el fallo medido.
    """
    from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "x86"
    mapping: QConfigMapping = get_default_qconfig_mapping("x86")
    if not transposed_int8:
        mapping = mapping.set_object_type(nn.ConvTranspose2d, None)
    if keep_float:
        for name, module in model.named_modules():
            if type(module).__name__ == "CBAM":
                mapping = mapping.set_module_name(name, None)
        for name in CAPAS_SALIDA:
            mapping = mapping.set_module_name(name, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(copy.deepcopy(model).eval(), mapping, example_inputs=(calibration[:1],))
        with torch.inference_mode():
            for start in range(0, len(calibration), 16):
                prepared(calibration[start:start + 16])
        quantized = convert_fx(prepared)
    quantized.instance_mode = model.instance_mode
    quantized.in_channels = getattr(model, "in_channels", 1)
    return quantized


def build(variant: str, base: nn.Module, calibration: torch.Tensor | None) -> nn.Module:
    if variant == "fp32":
        return base
    if variant == "fp16":
        return Precision(copy.deepcopy(base), torch.float16)
    if variant == "bf16":
        return Precision(copy.deepcopy(base), torch.bfloat16)
    return quantize_int8(base, calibration, keep_float=variant == "int8_cbam_fp32",
                         transposed_int8=variant == "int8_todo")


def size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1e6


@torch.inference_mode()
def latency_ms(model: nn.Module, slices: torch.Tensor, repeats: int, warmup: int = 3) -> dict:
    """Latencia de la red por corte (lote 1) en CPU; el NMS y el post-proceso no cambian."""
    for i in range(warmup):
        model(slices[i % len(slices)][None])
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        model(slices[i % len(slices)][None])
        times.append((time.perf_counter() - start) * 1000)
    return {"mediana_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


def evaluate_variant(model: nn.Module, cases: dict, index: dict, size: int, batch_size: int) -> dict:
    """Las mismas métricas de ``evaluate_volumes`` (2D) e ``infer_volume`` (3D y distancia)."""
    records_all, volumes = [], []
    for case_id, case in cases.items():
        records, pred = evaluate_case(model, case_id, case["images"], case["labels"], torch.device("cpu"),
                                      batch_size, keep_predictions=True)
        records_all.extend(records)
        volume = postprocess_3d(pred["semantic"], pred["primary_secondary"])
        nofill = postprocess_3d(pred["semantic"], pred["primary_secondary"], fill_unlabeled=False)
        spacing = cache_spacing_xyz(index[case_id], size)
        volumes.append({"volumen": summarize_volume(volume, case["labels"]),
                        "distancias": fragment_distance_table(nofill, case["labels"], spacing)})
    summary = summarize(records_all, "todo")
    out = {k: summary.get(k) for k in METRICAS}
    vol = aggregate(volumes)
    for key in ("dice_todos", "dice_principal", "dice_secundario", "error_distancia_mm_media",
                "correlacion_distancia"):
        out[key] = vol.get(key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTES), choices=VARIANTES)
    parser.add_argument("--tta", action="store_true", help="Espejo izquierda-derecha, como en el reporte de H6.")
    parser.add_argument("--calib-cases", type=int, default=8)
    parser.add_argument("--calib-per-case", type=int, default=16)
    parser.add_argument("--threads", type=int, default=None, help="Hilos de CPU (por defecto, los de torch).")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-metrics", action="store_true", help="Solo tamaño y latencia.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.split == "test":
        print("AVISO: test se usa una sola vez, para reportar la variante ya elegida en validación.")
    if args.threads:
        torch.set_num_threads(args.threads)
    cache_dir = Path(args.cache_dir)
    size = int(cache_dir.name.split("_")[1])
    base, _ = load_model([args.checkpoint], torch.device("cpu"), tta=False)
    canonical = bool(getattr(base, "canonical", False))
    calibration = None
    if any(v.startswith("int8") for v in args.variants):
        calibration = calibration_slices(cache_dir, Path(args.splits), args.calib_cases,
                                         args.calib_per_case, canonical)
        print(f"Calibración int8: {len(calibration)} cortes de entrenamiento", flush=True)

    case_ids = [str(c).zfill(3) for c in args.cases] if args.cases else cases_of_split(cache_dir, args.split)
    cases = {c: load_case_cache(cache_dir, c, canonical=canonical) for c in case_ids}
    index = read_index(cache_dir)
    bone = [c["images"][(c["labels"] != 0).any(axis=(1, 2))] for c in cases.values()]
    latency_slices = torch.from_numpy(from_uint8(np.concatenate(bone)[:: max(1, sum(map(len, bone)) // 60)]))[:, None]

    results = {}
    for variant in args.variants:
        started = time.time()
        try:
            model = build(variant, base, calibration).eval()
            with torch.inference_mode():
                model(latency_slices[:1])
        except Exception as error:   # p. ej. fp16 sin kernels en esta CPU: se reporta, no se inventa
            results[variant] = {"error": f"{type(error).__name__}: {error}"[:300]}
            print(f"{variant}: no se pudo ejecutar ({results[variant]['error']})", flush=True)
            continue
        row = {"tamano_mb": size_mb(model), **latency_ms(model, latency_slices, args.repeats)}
        if not args.skip_metrics:
            from evaluate_week10 import EnsembleTTA
            evaluated = EnsembleTTA([model], True) if args.tta else model
            row.update(evaluate_variant(evaluated, cases, index, size, args.batch_size))
        row["segundos"] = round(time.time() - started, 1)
        results[variant] = row
        print(f"{variant:15s} " + " ".join(f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)),
              flush=True)

    reference = results.get("fp32", {})
    for variant, row in results.items():
        if variant != "fp32" and "instance_dice_matched" in row and "instance_dice_matched" in reference:
            row["delta_dice_fragmento"] = row["instance_dice_matched"] - reference["instance_dice_matched"]
            row["cumple_regla_0_003"] = bool(row["delta_dice_fragmento"] >= -0.003)
    result = {"checkpoint": args.checkpoint, "split": args.split, "casos": case_ids, "tta": args.tta,
              "hilos_cpu": torch.get_num_threads(), "cpu": platform.processor() or platform.machine(),
              "torch": torch.__version__, "calibracion": {"casos_train": args.calib_cases,
                                                         "cortes_por_caso": args.calib_per_case},
              "variantes": results}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
