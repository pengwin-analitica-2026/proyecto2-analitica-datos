"""Etapa A en Kaggle: entrena Semana 10 durante exactamente 60 épocas.

El kernel parte del checkpoint físico 2.5D de Semana 9, no usa EMA ni parada
temprana y guarda por separado el mejor checkpoint y los pesos de la época 60.
Las evaluaciones de los puntos 5 a 9 se ejecutan en un segundo kernel para no
arriesgar el límite de tiempo de una única sesión de Kaggle.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


NOMBRE = "s10_kaggle_60ep_sin_early_stopping"
CODIGO = Path("/kaggle/temp/codigo")
ENTRADA = Path("/kaggle/input")
SALIDA = Path("/kaggle/working")
MEJOR = SALIDA / f"{NOMBRE}_best.pth"
ULTIMO = SALIDA / f"{NOMBRE}_epoch60.pth"
HISTORIAL = SALIDA / f"{NOMBRE}_historial.json"
LOG = SALIDA / f"{NOMBRE}_train.log"


def ejecutar(comando: list[str], log: Path | None = None) -> None:
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
    if log is not None:
        log.write_text("".join(lineas), encoding="utf-8")
    print(f"Finalizó con código {codigo} en {(time.time() - inicio) / 60:.1f} min", flush=True)
    if codigo != 0:
        raise RuntimeError(f"Falló el comando con código {codigo}")


def localizar_datos() -> tuple[Path, Path]:
    indices = [Path(p) for p in glob.glob(str(ENTRADA / "**" / "index.csv"), recursive=True)]
    if not indices:
        raise FileNotFoundError("No se encontró index.csv en los datasets montados.")
    for indice in indices:
        cache = indice.parent / "cache_384"
        candidatos = list(ENTRADA.rglob("week9_25d_rotacion_fisica_desde_cero_w32_e35_s42.pth"))
        if cache.is_dir() and len(list(cache.glob("*.npz"))) == 100 and candidatos:
            return indice.parent, candidatos[0]
    raise FileNotFoundError("El dataset no contiene cache_384 completo y el checkpoint de Semana 9.")


def verificar_checkpoint(path: Path, clase: str, epoch: int | None = None) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    no_finitos = sum(
        int((~torch.isfinite(valor)).sum())
        for valor in checkpoint["model"].values()
        if torch.is_tensor(valor) and valor.is_floating_point()
    )
    if no_finitos:
        raise RuntimeError(f"{path.name} contiene {no_finitos} valores no finitos.")
    if clase == "ultima_epoca":
        if checkpoint.get("checkpoint_kind") != clase:
            raise RuntimeError("El checkpoint final no está marcado como última época.")
        if int(checkpoint.get("epoch_completed", -1)) != int(epoch):
            raise RuntimeError("El checkpoint final no corresponde a la época solicitada.")


def guardar_csv(historial: dict) -> None:
    filas = historial["history"]
    if not filas:
        return
    columnas = sorted({clave for fila in filas for clave in fila})
    with (SALIDA / f"{NOMBRE}_historial.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_grafica(historial: dict) -> None:
    import matplotlib.pyplot as plt

    filas = historial["history"]
    epocas = [fila["epoch"] for fila in filas]
    figura, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
    ejes[0].plot(epocas, [fila["loss"] for fila in filas], marker="o")
    ejes[0].set(title="Pérdida de entrenamiento", xlabel="Época", ylabel="Loss")
    for clave, etiqueta in (
        ("segmentation_dice_macro", "Dice semántico"),
        ("detection_map50", "mAP50"),
        ("instance_dice_matched", "Dice instancias"),
    ):
        ejes[1].plot(epocas, [fila[clave] for fila in filas], marker="o", label=etiqueta)
    ejes[1].set(title="Validación durante entrenamiento", xlabel="Época", ylim=(0, 1))
    ejes[1].legend()
    figura.tight_layout()
    figura.savefig(SALIDA / f"{NOMBRE}_curvas.png", dpi=180)
    plt.close(figura)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    os.chdir(CODIGO)
    subprocess.run(["nvidia-smi"], check=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Este experimento requiere una GPU CUDA.")
    print("PyTorch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0), flush=True)

    cache_root, week9 = localizar_datos()
    print("Cache:", cache_root, "| checkpoint Semana 9:", week9, flush=True)

    comando = [
        sys.executable,
        "src/train_s10.py",
        "--cache-root", str(cache_root),
        "--size", "384",
        "--epochs", "60",
        "--lr", "0.002",
        "--base-channels", "32",
        "--cbam", "1",
        "--secondary-weight", "3",
        "--augment",
        "--cosine",
        "--batch-size", "2",
        "--per-epoch", "2500",
        "--eval-every", "5",
        "--context", "1",
        "--seed", "42",
        "--aug-geo", "1.0",
        "--aug-rot-deg", "35",
        "--canonical",
        "--aug-mm",
        "--ema", "0",
        "--early-stopping-patience", "0",
        "--checkpoint-metric", "instance_dice",
        "--week9-checkpoint", str(week9),
        "--freeze-week9-epochs", "5",
        "--output", str(MEJOR),
        "--last-output", str(ULTIMO),
        "--metrics-output", str(HISTORIAL),
    ]
    # Deliberadamente no se pasan --stop-dice, --stop-instance-dice,
    # --max-minutes ni --disable-cudnn.
    ejecutar(comando, LOG)

    historial = json.loads(HISTORIAL.read_text(encoding="utf-8"))
    if int(historial["epochs_completed"]) != 60:
        raise RuntimeError(f"Se esperaban 60 épocas y hubo {historial['epochs_completed']}.")
    if historial["stopped_early"] or historial["stop_reason"] is not None:
        raise RuntimeError(f"El entrenamiento se detuvo: {historial['stop_reason']}")
    verificar_checkpoint(MEJOR, "mejor")
    verificar_checkpoint(ULTIMO, "ultima_epoca", 60)
    guardar_csv(historial)
    guardar_grafica(historial)

    manifiesto = {
        "estado": "entrenamiento_60_epocas_completo",
        "epochs_completed": 60,
        "stopped_early": False,
        "mejor_checkpoint": MEJOR.name,
        "checkpoint_epoca_60": ULTIMO.name,
        "historial": HISTORIAL.name,
        "siguiente_etapa": "evaluacion_puntos_5_a_9",
    }
    (SALIDA / "etapa_a_entrenamiento.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("ENTRENAMIENTO DE 60 ÉPOCAS VERIFICADO", flush=True)


if __name__ == "__main__":
    main()
