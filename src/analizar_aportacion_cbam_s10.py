"""Ablación de inferencia de CBAM por cabeza para Semana 10.

Reutiliza la evaluación volumétrica con CBAM ya guardada por el notebook y
ejecuta solamente el pase adicional con CBAM puenteado. Nunca usa test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_cache import load_case_cache
from evaluate_volumes import (cache_spacing_yx, evaluate_case, load_model,
                              normalize_patient_rotation, summarize)


def modelos_base(model: nn.Module) -> list[nn.Module]:
    """Devuelve los modelos reales tanto con TTA/ensamble como sin él."""
    return list(model.models) if hasattr(model, "models") else [model]


def puentear_cbam(model: nn.Module) -> list[tuple[nn.Module, nn.Module, dict[str, nn.Module]]]:
    guardados = []
    for base in modelos_base(model):
        backbone = base.backbone
        if not getattr(backbone, "use_cbam", False):
            raise ValueError("El checkpoint no usa CBAM; no hay atención que analizar.")
        final = backbone.cbam
        intermedios = {k: modulo for k, modulo in backbone.cbam_intermedios.items()}
        backbone.cbam = nn.Identity()
        for nombre in list(backbone.cbam_intermedios.keys()):
            backbone.cbam_intermedios[nombre] = nn.Identity()
        guardados.append((backbone, final, intermedios))
    return guardados


def restaurar_cbam(guardados: list[tuple[nn.Module, nn.Module, dict[str, nn.Module]]]) -> None:
    for backbone, final, intermedios in guardados:
        backbone.cbam = final
        for nombre, modulo in intermedios.items():
            backbone.cbam_intermedios[nombre] = modulo


def ruta_evaluada(evaluacion: dict) -> list[Path]:
    valor = evaluacion.get("checkpoint", [])
    valores = valor if isinstance(valor, list) else [valor]
    return [Path(v).resolve() for v in valores]


def puntuaciones(resumen: dict) -> dict[str, float]:
    clasificacion = float(resumen["classification_f1_macro"])
    # Una caja por clase y sin refinamiento semántico: aísla la cabeza de detección.
    deteccion = float(resumen["detection_raw_one_map50"])
    semantica = float(resumen["segmentation_dice_macro"])
    instancias = float(resumen["instance_dice_matched"])
    segmentacion = (semantica + instancias) / 2.0
    general = (clasificacion + deteccion + segmentacion) / 3.0
    return {
        "clasificacion": clasificacion,
        "deteccion": deteccion,
        "segmentacion": segmentacion,
        "dice_semantico": semantica,
        "dice_instancias": instancias,
        "general": general,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--with-cbam-json", required=True,
                        help="Evaluación volumétrica de validación ya producida con CBAM.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    evaluacion = json.loads(Path(args.with_cbam_json).read_text(encoding="utf-8"))
    if evaluacion.get("split") != "val":
        raise ValueError("La aportación de CBAM solo se calibra en validación; test está prohibido.")
    if checkpoint_path not in ruta_evaluada(evaluacion):
        raise ValueError("La evaluación con CBAM pertenece a otro checkpoint.")

    case_ids = [str(c).zfill(3) for c in evaluacion["cases"]]
    confidence = float(evaluacion.get("confidence", 0.10))
    tta = bool(evaluacion.get("tta", False))
    two_pass = bool(evaluacion.get("two_pass", False))
    normalizar = bool(evaluacion.get("normalizar", False))
    resumen_con = evaluacion["subsets"]["todo"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model([str(checkpoint_path)], device, tta)
    cache_dir = Path(args.cache_dir)
    registros = []
    guardados = puentear_cbam(model)
    inicio = time.time()
    try:
        for numero, case_id in enumerate(case_ids, 1):
            case = load_case_cache(cache_dir, case_id,
                                   canonical=bool(getattr(model, "canonical", False)))
            if normalizar:
                case["images"], case["labels"], _ = normalize_patient_rotation(
                    case["images"], case["labels"], *cache_spacing_yx(case))
            filas, _ = evaluate_case(
                model, case_id, case["images"], case["labels"], device,
                args.batch_size, confidence, two_pass, keep_predictions=False)
            registros.extend(filas)
            print(f"  sin CBAM {numero}/{len(case_ids)}: caso {case_id}", flush=True)
    finally:
        restaurar_cbam(guardados)

    resumen_sin = summarize(registros, "todo")
    con = puntuaciones(resumen_con)
    sin = puntuaciones(resumen_sin)
    aporte = {nombre: con[nombre] - sin[nombre] for nombre in con}
    caida = 100.0 * aporte["general"] / max(con["general"], 1e-12)

    print("\n=== Aportación general de CBAM — Semana 10 ===")
    for etiqueta, clave in (("clasificación", "clasificacion"),
                            ("detección", "deteccion"),
                            ("segmentación", "segmentacion")):
        print(f"\nCabeza de {etiqueta}:")
        print(f"  con CBAM = {con[clave]:.4f}")
        print(f"  sin CBAM = {sin[clave]:.4f}")
        print(f"  aportación = {aporte[clave]:+.4f}")
    print(f"\nScore general con CBAM: {con['general']:.4f}")
    print(f"Score general sin CBAM: {sin['general']:.4f}")
    print(f"Aportación general:      {aporte['general']:+.4f}")
    print(f"Caída al retirar CBAM:   {caida:.1f} %")
    print("\nDetalle de segmentación:")
    print(f"  Dice semántico:  {con['dice_semantico']:.4f} -> {sin['dice_semantico']:.4f} "
          f"({aporte['dice_semantico']:+.4f})")
    print(f"  Dice instancias: {con['dice_instancias']:.4f} -> {sin['dice_instancias']:.4f} "
          f"({aporte['dice_instancias']:+.4f})")
    print("\nNota: es una ablación de inferencia del mismo checkpoint; no sustituye "
          "entrenar un control desde cero sin CBAM.")

    resultado = {
        "metodo": "ablacion_inferencia_mismo_checkpoint",
        "checkpoint": str(checkpoint_path), "split": "val", "cases": case_ids,
        "confidence": confidence, "tta": tta, "two_pass": two_pass,
        "normalizar": normalizar, "seconds_without_cbam": round(time.time() - inicio, 1),
        "definicion_score": {
            "clasificacion": "classification_f1_macro",
            "deteccion": "detection_raw_one_map50",
            "segmentacion": "promedio(segmentation_dice_macro, instance_dice_matched)",
            "general": "promedio(clasificacion, deteccion, segmentacion)",
        },
        "con_cbam": con, "sin_cbam": sin, "aportacion": aporte,
        "caida_al_retirar_cbam_pct": caida,
        "metricas_completas_sin_cbam": resumen_sin,
        "advertencia": "Mide dependencia en inferencia, no ganancia causal de entrenamiento.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print("JSON:", output)


if __name__ == "__main__":
    main()
