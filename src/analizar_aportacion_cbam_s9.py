"""Mide cuánto depende el detector de Semana 9 del CBAM ya entrenado.

La arquitectura tiene una sola cabeza de detección con tres componentes:
objectness, regresión de caja y clasificación anatómica. Este script compara
el mismo checkpoint con CBAM activo y con sus módulos CBAM puenteados por
Identity. Es una ablación de inferencia (sensibilidad del modelo entrenado),
no reemplaza el experimento causal de entrenar otro modelo desde cero sin CBAM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from week9_detection import (boxes_from_label, decode_predictions, detection_loss,
                              detection_model_from_checkpoint, labels_to_grid,
                              match_detections)


def evaluar(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
            device: torch.device, confidence: float, batch_size: int = 16) -> dict[str, float]:
    model.eval()
    conteos = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n_gt": 0.0,
               "n_pred": 0.0, "iou_suma": 0.0}
    perdidas = {"total": 0.0, "objectness": 0.0, "box": 0.0,
                "classification": 0.0}
    muestras = 0

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size].to(device)
            y = labels[start:start + batch_size]
            pred = model(x)
            target = labels_to_grid(y).to(device)
            _, partes = detection_loss(pred, target)
            n = len(x)
            muestras += n
            for nombre in perdidas:
                perdidas[nombre] += partes[nombre] * n

            predicciones = decode_predictions(pred, confidence=confidence)
            verdades = boxes_from_label(y)
            for predicha, real in zip(predicciones, verdades):
                for nombre, valor in match_detections(predicha, real, 0.5).items():
                    conteos[nombre] += valor

    tp, fp, fn = conteos["tp"], conteos["fp"], conteos["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou_medio": conteos["iou_suma"] / max(conteos["n_gt"], 1),
        "falsos_positivos": fp,
        "n_gt": conteos["n_gt"],
        "n_pred": conteos["n_pred"],
        **{f"loss_{k}": v / max(muestras, 1) for k, v in perdidas.items()},
    }


def puentear_cbam(model: nn.Module) -> tuple[nn.Module, dict[str, nn.Module]]:
    """Sustituye temporalmente los CBAM por Identity y devuelve los originales."""
    backbone = model.backbone
    original_final = backbone.cbam
    originales_intermedios = {k: modulo for k, modulo in backbone.cbam_intermedios.items()}
    backbone.cbam = nn.Identity()
    for nombre in list(backbone.cbam_intermedios.keys()):
        backbone.cbam_intermedios[nombre] = nn.Identity()
    return original_final, originales_intermedios


def restaurar_cbam(model: nn.Module, original_final: nn.Module,
                   originales_intermedios: dict[str, nn.Module]) -> None:
    backbone = model.backbone
    backbone.cbam = original_final
    for nombre, modulo in originales_intermedios.items():
        backbone.cbam_intermedios[nombre] = modulo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--output", default=None,
                        help="JSON de salida; por defecto se crea junto al checkpoint.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint.get("use_cbam", True):
        raise ValueError("El checkpoint no contiene CBAM; no hay atención que puentear.")

    data = np.load(args.cache)
    val = np.asarray(data["split_names"] == "val")
    images = torch.from_numpy(data["images"][val].astype(np.float32))
    if images.ndim == 3:
        images = images.unsqueeze(1)
    labels = torch.from_numpy(data["labels"][val].astype(np.int64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint.get("disable_cudnn", False):
        torch.backends.cudnn.enabled = False
    model = detection_model_from_checkpoint(checkpoint).to(device)

    con_cbam = evaluar(model, images, labels, device, args.confidence)
    original_final, originales_intermedios = puentear_cbam(model)
    try:
        sin_cbam = evaluar(model, images, labels, device, args.confidence)
    finally:
        restaurar_cbam(model, original_final, originales_intermedios)

    # Positivo significa que CBAM ayuda: con-sin para métricas y sin-con para pérdidas.
    mayores_mejor = ("f1", "precision", "recall", "iou_medio")
    menores_mejor = ("loss_total", "loss_objectness", "loss_box", "loss_classification")
    aporte = {k: con_cbam[k] - sin_cbam[k] for k in mayores_mejor}
    aporte.update({k: sin_cbam[k] - con_cbam[k] for k in menores_mejor})

    etiquetas = {
        "f1": "F1 detección", "precision": "Precisión", "recall": "Recall",
        "iou_medio": "IoU medio", "loss_total": "Pérdida total",
        "loss_objectness": "Salida objeto", "loss_box": "Salida caja",
        "loss_classification": "Salida clasificación",
    }
    print("\n=== Aportación de CBAM — Semana 9 (validación) ===")
    print("Cabeza: detección por grid | componentes: objeto + caja + clasificación")
    print(f"Confianza={args.confidence:.2f} | muestras={len(images)} | dispositivo={device}")
    print("Aportación positiva = resultado favorable al mantener CBAM activo.\n")
    print(f"{'Componente':<23} {'Con CBAM':>12} {'Sin CBAM':>12} {'Aportación':>12}")
    print("-" * 63)
    for nombre in (*mayores_mejor, *menores_mejor):
        print(f"{etiquetas[nombre]:<23} {con_cbam[nombre]:>12.4f} "
              f"{sin_cbam[nombre]:>12.4f} {aporte[nombre]:>+12.4f}")

    delta_f1 = aporte["f1"]
    caida_sin_cbam = 100.0 * delta_f1 / max(con_cbam["f1"], 1e-12)
    factor_f1 = con_cbam["f1"] / max(sin_cbam["f1"], 1e-12)
    print("\n=== Aportación general en la cabeza de detección ===")
    print(f"CBAM aporta {delta_f1:+.4f} de F1 ({delta_f1 * 100:+.2f} puntos porcentuales).")
    print(f"Al puentearlo, el F1 cae {caida_sin_cbam:.1f}% respecto al modelo completo "
          f"({con_cbam['f1']:.4f} -> {sin_cbam['f1']:.4f}).")
    print(f"Con CBAM, el F1 es {factor_f1:.2f} veces el obtenido al puentear la atención.")
    print("\nNota: esto mide dependencia por ablación en inferencia. Para afirmar ganancia causal,\n"
          "entrena un control idéntico desde cero con --no-cbam y compara validación.")

    output = (Path(args.output) if args.output else
              checkpoint_path.with_name(checkpoint_path.stem + "_cbam_aportacion.json"))
    resultado = {
        "metodo": "ablacion_inferencia_mismo_checkpoint",
        "checkpoint": str(checkpoint_path), "cache": str(args.cache),
        "confidence": args.confidence, "muestras_validacion": len(images),
        "cabeza": "deteccion_grid",
        "componentes": ["objectness", "box", "classification"],
        "con_cbam": con_cbam, "sin_cbam": sin_cbam, "aportacion": aporte,
        "aportacion_general_cabeza_deteccion": {
            "metrica_principal": "f1",
            "delta_f1": delta_f1,
            "puntos_porcentuales_f1": delta_f1 * 100.0,
            "caida_relativa_al_puentear_cbam_pct": caida_sin_cbam,
            "factor_f1_con_vs_sin_cbam": factor_f1,
        },
        "advertencia": "No sustituye un entrenamiento controlado desde cero sin CBAM.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print("JSON:", output)


if __name__ == "__main__":
    main()
