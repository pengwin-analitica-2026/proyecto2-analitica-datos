"""Etapa B en Kaggle: completa los puntos 5 a 9 de Semana 10.

Selecciona entre el mejor checkpoint del entrenamiento y la época 60 usando
solo validación. Después congela esa decisión, evalúa test una vez, audita la
rotación, publica el modelo, mide la aportación de CBAM y calcula HD95/ASSD.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


CODIGO = Path("/kaggle/temp/codigo")
ENTRADA = Path("/kaggle/input")
SALIDA = Path("/kaggle/working")
BATCH_SIZE = 2


def ejecutar(comando: list[str], titulo: str, log: Path) -> None:
    print(f"\n=== {titulo} ===", flush=True)
    print("$", " ".join(map(str, comando)), flush=True)
    inicio = time.time()
    lineas: list[str] = []
    with subprocess.Popen(
        comando,
        cwd=CODIGO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as proceso:
        assert proceso.stdout is not None
        for linea in proceso.stdout:
            print(linea, end="", flush=True)
            lineas.append(linea)
        codigo = proceso.wait()
    log.write_text("".join(lineas), encoding="utf-8")
    if codigo != 0:
        raise RuntimeError(f"{titulo} falló con código {codigo}.")
    print(f"=== {titulo}: {(time.time() - inicio) / 60:.1f} min ===", flush=True)


def unico(nombre: str) -> Path:
    candidatos = list(ENTRADA.rglob(nombre))
    if len(candidatos) != 1:
        raise FileNotFoundError(f"Se esperaba un único {nombre}; encontrados: {candidatos}")
    return candidatos[0]


def evaluar(checkpoint: Path, cache: Path, split: str, salida: Path, titulo: str) -> None:
    ejecutar(
        [
            sys.executable,
            "src/evaluate_volumes.py",
            "--checkpoint", str(checkpoint),
            "--cache-dir", str(cache),
            "--split", split,
            "--batch-size", str(BATCH_SIZE),
            "--tta",
            "--output", str(salida),
        ],
        titulo,
        salida.with_suffix(".log"),
    )


def metrica_seleccion(path: Path) -> float:
    datos = json.loads(path.read_text(encoding="utf-8"))
    return float(datos["subsets"]["todo"]["instance_dice_matched"])


def comprobar_trazabilidad(checkpoint: Path, historial: dict) -> None:
    if int(historial["epochs_completed"]) != 60 or historial["stopped_early"]:
        raise RuntimeError("La etapa A no demuestra 60 épocas completas.")
    modelo = torch.load(checkpoint, map_location="cpu", weights_only=False)
    transferencia = modelo.get("week9_transfer")
    if not transferencia or transferencia.get("backbone_tensors", 0) <= 0:
        raise RuntimeError("No se puede demostrar la transferencia del backbone de Semana 9.")
    if transferencia.get("detection_tensors", 0) <= 0:
        raise RuntimeError("No se puede demostrar la transferencia del detector de Semana 9.")
    if not transferencia.get("aug_mm") or float(transferencia.get("aug_rot_deg", -1)) != 35.0:
        raise RuntimeError("El checkpoint de Semana 9 no registra la rotación física esperada.")
    no_finitos = sum(
        int((~torch.isfinite(valor)).sum())
        for valor in modelo["model"].values()
        if torch.is_tensor(valor) and valor.is_floating_point()
    )
    if no_finitos:
        raise RuntimeError(f"El checkpoint contiene {no_finitos} valores no finitos.")


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    os.chdir(CODIGO)
    subprocess.run(["nvidia-smi"], check=False)
    if not torch.cuda.is_available():
        raise RuntimeError("La evaluación completa requiere GPU CUDA.")

    indice = unico("index.csv")
    cache = indice.parent / "cache_384"
    if len(list(cache.glob("*.npz"))) != 100:
        raise RuntimeError("El caché montado no contiene los 100 casos.")
    mejor = unico("s10_kaggle_60ep_sin_early_stopping_best.pth")
    ultimo = unico("s10_kaggle_60ep_sin_early_stopping_epoch60.pth")
    historial_path = unico("s10_kaggle_60ep_sin_early_stopping_historial.json")
    historial = json.loads(historial_path.read_text(encoding="utf-8"))
    comprobar_trazabilidad(mejor, historial)
    comprobar_trazabilidad(ultimo, historial)
    print("Punto 5: trazabilidad y 60 épocas verificadas.", flush=True)

    # Punto 6: ambos candidatos se comparan únicamente en validación.
    val_mejor = SALIDA / "candidato_best_val.json"
    val_ultimo = SALIDA / "candidato_epoch60_val.json"
    evaluar(mejor, cache, "val", val_mejor, "Validación volumétrica del mejor checkpoint")
    evaluar(ultimo, cache, "val", val_ultimo, "Validación volumétrica de la época 60")
    candidatos = [(mejor, val_mejor), (ultimo, val_ultimo)]
    seleccionado, validacion_temporal = max(candidatos, key=lambda item: metrica_seleccion(item[1]))
    print(
        "Checkpoint seleccionado solo con validación:", seleccionado.name,
        "| Dice instancias:", f"{metrica_seleccion(validacion_temporal):.4f}",
        flush=True,
    )

    modelo_final = SALIDA / "modelo_final_semana10.pth"
    shutil.copy2(seleccionado, modelo_final)
    validacion = SALIDA / "modelo_final_semana10_val.json"
    # Reescribe la ruta del checkpoint mediante una evaluación limpia del archivo publicado.
    evaluar(modelo_final, cache, "val", validacion, "Validación oficial del modelo publicado")
    prueba = SALIDA / "modelo_final_semana10_test.json"
    evaluar(modelo_final, cache, "test", prueba, "Test oficial congelado")

    # Auditoría de orientación y rotación física, siempre solo en validación.
    auditoria = SALIDA / "auditoria_rotacion_fisica"
    auditoria.mkdir(exist_ok=True)
    experimentos = [
        ("limpio_normalizado", None, True),
        ("rotmm30_sin_corregir", "rotmm=30", False),
        ("rotmm30_corregido", "rotmm=30", True),
        ("rotmm60_sin_corregir", "rotmm=60", False),
        ("rotmm60_corregido", "rotmm=60", True),
    ]
    for nombre, stress, normalizar in experimentos:
        salida = auditoria / f"{nombre}.json"
        comando = [
            sys.executable,
            "src/evaluate_volumes.py",
            "--checkpoint", str(modelo_final),
            "--cache-dir", str(cache),
            "--split", "val",
            "--batch-size", str(BATCH_SIZE),
            "--tta",
            "--output", str(salida),
            "--normalizar" if normalizar else "--no-normalizar",
        ]
        if stress:
            comando.extend(["--stress", stress])
        ejecutar(comando, f"Auditoría de rotación: {nombre}", salida.with_suffix(".log"))

    # Punto 7: publicación reproducible con huella del archivo.
    sha256 = hashlib.sha256(modelo_final.read_bytes()).hexdigest()
    manifiesto = {
        "modelo_final": str(modelo_final),
        "checkpoint_seleccionado": seleccionado.name,
        "criterio_seleccion": "instance_dice_matched de validación volumétrica (subconjunto todo)",
        "candidatos_validacion": {
            mejor.name: metrica_seleccion(val_mejor),
            ultimo.name: metrica_seleccion(val_ultimo),
        },
        "sha256": sha256,
        "bytes": modelo_final.stat().st_size,
        "historial_entrenamiento": historial_path.name,
        "evaluacion_validacion": validacion.name,
        "evaluacion_prueba": prueba.name,
    }
    (SALIDA / "modelo_final_semana10.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Punto 7: modelo final publicado. SHA-256:", sha256, flush=True)

    # Punto 8: aportación de CBAM por cabeza, sin tocar test.
    ejecutar(
        [
            sys.executable,
            "src/analizar_aportacion_cbam_s10.py",
            "--checkpoint", str(modelo_final),
            "--cache-dir", str(cache),
            "--with-cbam-json", str(validacion),
            "--batch-size", str(BATCH_SIZE),
            "--output", str(SALIDA / "modelo_final_semana10_cbam_aportacion.json"),
        ],
        "Punto 8: aportación de CBAM por cabeza",
        SALIDA / "modelo_final_semana10_cbam_aportacion.log",
    )

    # Punto 9: HD95 y ASSD por fragmento en validación y test.
    for split in ("val", "test"):
        base = SALIDA / f"modelo_final_semana10_hd95_assd_{split}"
        ejecutar(
            [
                sys.executable,
                "src/evaluar_metricas_superficie.py",
                "--checkpoint", str(modelo_final),
                "--cache-dir", str(cache),
                "--split", split,
                "--batch-size", str(BATCH_SIZE),
                "--tta",
                "--output", str(base.with_suffix(".json")),
                "--csv-output", str(base.with_suffix(".csv")),
            ],
            f"Punto 9: HD95 y ASSD — {split}",
            base.with_suffix(".log"),
        )

    resumen = {
        "estado": "puntos_1_a_9_completos",
        "entrenamiento_epocas": 60,
        "stopped_early": False,
        "modelo_final": modelo_final.name,
        "manifiesto": "modelo_final_semana10.json",
        "validacion": validacion.name,
        "test": prueba.name,
        "cbam": "modelo_final_semana10_cbam_aportacion.json",
        "hd95_assd_val": "modelo_final_semana10_hd95_assd_val.json",
        "hd95_assd_test": "modelo_final_semana10_hd95_assd_test.json",
    }
    (SALIDA / "EXPERIMENTO_COMPLETO_PUNTO_9.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("EXPERIMENTO COMPLETO HASTA EL PUNTO 9", flush=True)


if __name__ == "__main__":
    main()
