"""Entrenamiento multitarea de la Semana 10 sobre los cachés por caso (volumen completo).

Diferencias con ``train_week10_multitask.py``:

- lee ``cache_<size>[_pad]/<caso>.npz`` (``build_case_cache.py``), no un npz único de cortes;
- cada época muestrea cortes distintos de los 70 casos de entrenamiento (todos los cortes
  con hueso más un margen sin hueso), en vez de repetir los mismos 12 por caso;
- el CBAM es configurable: ``--cbam none|1|3`` y ``--learnable-gamma``;
- permite ``--mode pad`` (conserva la proporción de los volúmenes no cuadrados);
- límite de tiempo ``--max-minutes`` para respetar las cuotas de Kaggle/Colab: al llegar
  guarda el mejor punto hasta ese momento.

Opciones de la exploración de la Semana 10 (todas apagadas por defecto, así que sin
ellas el comportamiento es el de B1):

- ``--context K``: contexto 2.5D, los K cortes vecinos de cada lado entran como canales
  extra (en el borde del caso se repite el último corte);
- ``--focal-gamma`` y ``--tversky-beta``: pérdidas para fragmentos pequeños;
- ``--deep-supervision-weight``: salidas auxiliares en el decodificador;
- ``--ema``: promedio móvil exponencial de los pesos (se evalúa y guarda el promedio);
- ``--normalizar``: enderezado pélvico conservador antes de formar los cortes;
- ``--aug-mm``: aplica la rotación aleatoria de ``--aug-geo`` en coordenadas físicas;
- ``--secondary-oversample``: muestrea más los cortes con fragmentos secundarios;
- ``--backbone-init``: pesos del backbone preentrenados con ``pretrain_backbone.py``.

La evaluación durante el entrenamiento usa un subconjunto fijo de validación (un corte
de cada ``--val-stride``). La cifra definitiva sale de ``evaluate_volumes.py`` sobre el
volumen completo.

Uso:

    python src/train_s10.py --cache-root data/cache/s10 --size 384 --epochs 40 \\
        --cbam 1 --augment --cosine --output outputs/s10/b0_s42.pth
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_cache import cache_dir_name, from_uint8, load_case_cache
from aumento_fisico import random_affine
from orientacion import estimate_pelvis_correction, rotate_inplane_physical
from train_week10_multitask import evaluate, seed_everything
from week9_detection import labels_to_grid
from week10_multitask import (INSTANCE_MODES, MultiTaskLossWeights, PengwinMultiTaskModel,
                              flip_sagittal, fracture_distance_weight, multitask_loss)


def read_split(cache_root: Path) -> dict[str, list[str]]:
    with open(cache_root / "index.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for row in rows:
        out.setdefault(row["split"], []).append(row["case_id"])
    return out


def load_slices(cache_dir: Path, case_ids: list[str], margin: int, stride: int = 1,
                keep_negatives: bool = True, canonical: bool = False,
                normalize_rotation: bool = False) -> dict[str, np.ndarray]:
    """Carga los cortes de interés de cada caso: extensión ósea ± ``margin``.

    Devuelve arrays uint8 concatenados y un vector ``has_bone`` para muestrear.
    """
    images, labels, bone, case_of, lows, highs, secondary, spacings = [], [], [], [], [], [], [], []
    normalizations: dict[str, dict] = {}
    total = 0
    for case_number, case_id in enumerate(case_ids, 1):
        data = load_case_cache(cache_dir, case_id, canonical=canonical)
        sx, sy = (float(v) for v in data.get("spacing_xyz", (1.0, 1.0, 1.0))[:2])
        orig_shape = data.get("orig_shape", data["images"].shape)
        mm_y = sy * int(orig_shape[-2]) / data["images"].shape[-2]
        mm_x = sx * int(orig_shape[-1]) / data["images"].shape[-1]
        info = None
        if normalize_rotation:
            info = estimate_pelvis_correction(data["images"], mm_y, mm_x)
            normalizations[str(case_id).zfill(3)] = info
            if case_number % 10 == 0 or case_number == len(case_ids):
                accepted = sum(bool(row.get("accepted")) for row in normalizations.values())
                print(f"  normalización: {case_number}/{len(case_ids)} casos, "
                      f"{accepted} correcciones aceptadas", flush=True)
        has_bone = (data["labels"] != 0).any(axis=(1, 2))
        where = np.flatnonzero(has_bone)
        if len(where) == 0:
            continue
        low, high = max(int(where[0]) - margin, 0), min(int(where[-1]) + margin + 1, len(has_bone))
        picks = np.arange(low, high, stride)
        if not keep_negatives:
            picks = picks[has_bone[picks]]
        selected_images, selected_labels = data["images"][picks], data["labels"][picks]
        if info is not None and info["accepted"]:
            # El giro no modifica z: remuestrear solo la extensión que realmente se entrena evita
            # rotar cientos de cortes de aire y reduce varios minutos de carga.
            selected_images = rotate_inplane_physical(
                selected_images, info["angle_degrees"], mm_y, mm_x, order=1)
            selected_labels = rotate_inplane_physical(
                selected_labels, info["angle_degrees"], mm_y, mm_x, order=0)
            selected_images = np.clip(np.rint(selected_images), 0, 255).astype(np.uint8)
            selected_labels = selected_labels.astype(np.uint8, copy=False)
        images.append(selected_images); labels.append(selected_labels)
        bone.append(has_bone[picks]); case_of.extend([case_id] * len(picks))
        lab = selected_labels
        secondary.append(((lab % 10 != 1) & (lab > 0)).any(axis=(1, 2)))  # IDs 2..10, 12..20, 22..30
        spacings.append(np.tile(np.asarray([mm_y, mm_x], dtype=np.float32), (len(picks), 1)))
        lows.append(np.full(len(picks), total)); highs.append(np.full(len(picks), total + len(picks) - 1))
        total += len(picks)
    return {"images": np.concatenate(images), "labels": np.concatenate(labels),
            "has_bone": np.concatenate(bone), "case": np.asarray(case_of),
            "has_secondary": np.concatenate(secondary),
            "spacing_yx": np.concatenate(spacings), "normalization": normalizations,
            "case_lo": np.concatenate(lows), "case_hi": np.concatenate(highs)}


def to_batch(images_u8: np.ndarray, labels_u8: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Imágenes [N,H,W] (un canal) o [N,C,H,W] (con contexto) → tensores float/int64."""
    x = torch.from_numpy(from_uint8(images_u8))
    if x.ndim == 3:
        x = x.unsqueeze(1)
    y = torch.from_numpy(labels_u8.astype(np.int64))
    return x, y


def load_week9_initialization(model: torch.nn.Module, checkpoint_path: str | Path,
                              device: torch.device) -> dict:
    """Carga estrictamente el backbone, CBAM y detector producidos por Semana 9.

    Semana 10 agrega las cabezas de clasificación y segmentación desde cero. Se
    rechaza el checkpoint si el ancho o la cantidad de canales no coinciden, para
    impedir que una corrida que debía ser transferida continúe desde pesos aleatorios.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint de Semana 9: {path}. Ejecuta primero el notebook de Semana 9."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"El checkpoint de Semana 9 no contiene un state_dict 'model': {path}")

    expected_channels = int(getattr(model, "in_channels", 1))
    checkpoint_channels = int(checkpoint.get("in_channels", 1))
    if checkpoint_channels != expected_channels:
        raise ValueError(
            f"Canales incompatibles: Semana 9={checkpoint_channels}, Semana 10={expected_channels}. "
            "Ambos notebooks deben usar el mismo CONTEXT."
        )
    checkpoint_width = int(checkpoint.get("base_channels", -1))
    model_width = int(model.backbone.out_channels // 8)
    if checkpoint_width != model_width:
        raise ValueError(
            f"Ancho incompatible: Semana 9={checkpoint_width}, Semana 10={model_width}."
        )

    current = model.state_dict()
    prefixes = ("backbone.", "detection_head.")
    inherited = {key: value for key, value in state.items() if key.startswith(prefixes)}
    shape_errors = [key for key, value in inherited.items()
                    if key not in current or tuple(value.shape) != tuple(current[key].shape)]
    if shape_errors:
        raise ValueError(f"Tensores incompatibles en checkpoint Semana 9: {shape_errors}")
    backbone_keys = [key for key in inherited if key.startswith("backbone.")]
    detector_keys = [key for key in inherited if key.startswith("detection_head.")]
    if not backbone_keys or not detector_keys:
        raise ValueError(
            "El checkpoint no contiene backbone y detection_head; no se permite continuar desde cero."
        )
    model.load_state_dict(inherited, strict=False)
    print(f"Transferencia Semana 9: backbone={len(backbone_keys)} tensores, "
          f"detector={len(detector_keys)} tensores | {path}", flush=True)
    return {
        "path": str(path),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_f1": checkpoint.get("best_val_f1"),
        "backbone_tensors": len(backbone_keys),
        "detection_tensors": len(detector_keys),
        "in_channels": checkpoint_channels,
        "canonical": bool(checkpoint.get("canonical", False)),
        "aug_geo": float(checkpoint.get("aug_geo", 0.0)),
        "aug_rot_deg": float(checkpoint.get("aug_rot_deg", 0.0)),
        "aug_mm": bool(checkpoint.get("aug_mm", False)),
    }


def random_lowres(x: torch.Tensor, probability: float, max_factor: float = 2.6) -> torch.Tensor:
    """Simula cortes de baja resolución: reduce por un factor distinto en cada eje y vuelve a ampliar.

    Algunos volúmenes vienen con pocas filas o espaciado grueso y el caché los amplía hasta 2.5
    veces; se ven borrosos. Es el aumento de "baja resolución simulada" de nnU-Net.
    """
    height, width = x.shape[-2:]
    for i in torch.nonzero(torch.rand(x.shape[0]) < probability).flatten().tolist():
        fy = 1.0 + torch.rand(1).item() * (max_factor - 1.0)
        fx = 1.0 + torch.rand(1).item() * (max_factor - 1.0) * 0.5
        small = torch.nn.functional.interpolate(x[i:i + 1], size=(max(int(height / fy), 8), max(int(width / fx), 8)),
                                                mode="bilinear", align_corners=False)
        x[i:i + 1] = torch.nn.functional.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
    return x


def gather_context(data: dict, idx: np.ndarray, context: int) -> np.ndarray:
    """Cortes ``idx`` con ``context`` vecinos por lado como canales: [N, 2*context+1, H, W].

    El corte central es el canal ``context``. Los vecinos se limitan al rango de cortes
    cargados del mismo caso, así nunca se mezclan casos distintos.
    """
    if context <= 0:
        return data["images"][idx]
    lo, hi = data["case_lo"][idx], data["case_hi"][idx]
    return np.stack([data["images"][np.clip(idx + d, lo, hi)]
                     for d in range(-context, context + 1)], axis=1)


def epoch_indices(has_bone: np.ndarray, rng: np.random.Generator, neg_fraction: float,
                  per_epoch: int | None, has_secondary: np.ndarray | None = None,
                  oversample: float = 1.0) -> np.ndarray:
    """Todos los cortes con hueso (o ``per_epoch``) más una fracción de cortes sin hueso.

    Con ``oversample`` > 1 los cortes con fragmentos secundarios se eligen con esa
    probabilidad relativa (solo tiene efecto si ``per_epoch`` submuestrea).
    """
    bone_idx = np.flatnonzero(has_bone)
    neg_idx = np.flatnonzero(~has_bone)
    if per_epoch is not None and per_epoch < len(bone_idx):
        probability = None
        if has_secondary is not None and oversample > 1.0:
            probability = np.where(has_secondary[bone_idx], oversample, 1.0)
            probability = probability / probability.sum()
        bone_idx = rng.choice(bone_idx, per_epoch, replace=False, p=probability)
    n_neg = min(int(round(len(bone_idx) * neg_fraction)), len(neg_idx))
    chosen = np.concatenate([bone_idx, rng.choice(neg_idx, n_neg, replace=False)]) if n_neg else bone_idx
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-root", default="data/cache/s10")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--mode", choices=["stretch", "pad"], default="stretch")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--cbam", choices=["none", "1", "3"], default="1")
    parser.add_argument("--disable-cudnn", action="store_true",
                        help="Desactiva cuDNN; evita NaN FP16 observados en GTX 1650 con el modelo ancho 32.")
    parser.add_argument("--learnable-gamma", action="store_true")
    parser.add_argument("--instance-mode", choices=sorted(INSTANCE_MODES), default="primary")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--margin", type=int, default=10, help="Cortes sin hueso a cada lado de la extensión ósea.")
    parser.add_argument("--neg-fraction", type=float, default=0.15,
                        help="Cortes sin hueso por cada corte con hueso en cada época.")
    parser.add_argument("--per-epoch", type=int, default=None,
                        help="Cortes con hueso por época (por defecto, todos).")
    parser.add_argument("--val-stride", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Evaluaciones sin mejora suficiente del Dice antes de detener (0 = apagado).")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.005,
                        help="Mejora minima del Dice para reiniciar la paciencia.")
    parser.add_argument("--early-stopping-min-epochs", type=int, default=0,
                        help="Minimo de epocas antes de permitir la parada temprana.")
    parser.add_argument("--early-stopping-metric", choices=("semantic_dice", "instance_dice", "score"),
                        default="semantic_dice",
                        help="Metrica de validacion cuya mejora reinicia la paciencia.")
    parser.add_argument("--stop-dice", type=float, default=None,
                        help="Detiene al alcanzar este Dice semantico de validacion, despues del minimo de epocas.")
    parser.add_argument("--stop-instance-dice", type=float, default=None,
                        help="Objetivo de Dice de instancias; si se combina con --stop-dice deben cumplirse ambos.")
    parser.add_argument("--checkpoint-metric", choices=("score", "semantic_dice", "instance_dice"),
                        default="score", help="Metrica prioritaria para seleccionar el mejor checkpoint.")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None,
                        help="Checkpoint de Semana 10: carga todos los pesos e inicia una nueva fase de fine-tuning.")
    parser.add_argument("--week9-checkpoint", default=None,
                        help="Checkpoint 2.5D de Semana 9: carga backbone, CBAM y detector.")
    parser.add_argument("--freeze-week9-epochs", type=int, default=0,
                        help="Épocas iniciales con backbone y detector heredados congelados.")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--output", default="outputs/s10/modelo.pth")
    parser.add_argument(
        "--last-output",
        default=None,
        help=("Guarda además los pesos de la última época ejecutada. El archivo de "
              "--output sigue siendo el mejor checkpoint de validación."),
    )
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--fracture-weight", type=float, default=0.0)
    parser.add_argument("--secondary-weight", type=float, default=1.0)
    parser.add_argument("--detection-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=2.0)
    parser.add_argument("--semantic-dice-weight", type=float, default=1.0)
    parser.add_argument("--instance-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--instance-dice-weight", type=float, default=1.0)
    parser.add_argument("--context", type=int, default=0,
                        help="Cortes vecinos por lado como canales extra (2.5D).")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--tversky-beta", type=float, default=0.0)
    parser.add_argument("--deep-supervision-weight", type=float, default=0.0)
    parser.add_argument("--ema", type=float, default=0.0, help="Decaimiento del promedio móvil (0 = apagado).")
    parser.add_argument("--secondary-oversample", type=float, default=1.0)
    parser.add_argument("--canonical", action="store_true",
                        help="Lleva los 34 casos guardados en LPI a la orientación común (orientacion.py).")
    parser.add_argument("--normalizar", action="store_true",
                        help="Endereza por simetría pélvica cada caso si supera los controles de confianza.")
    parser.add_argument("--aug-geo", type=float, default=0.0,
                        help="Fuerza del aumento geométrico (escala, proporción, rotación, traslación); 0 = apagado.")
    parser.add_argument("--aug-rot-deg", type=float, default=12.0,
                        help="Amplitud de la rotación del aumento geométrico, en grados (con --aug-geo).")
    parser.add_argument("--aug-mm", action="store_true",
                        help="Interpreta --aug-rot-deg en el espacio físico usando el spacing de cada caso.")
    parser.add_argument("--aug-rot-wide", type=float, default=0.0,
                        help="Fracción de muestras con rotación libre de +-180 grados (requiere --aug-geo).")
    parser.add_argument("--aug-lowres", type=float, default=0.0,
                        help="Probabilidad de simular baja resolución por muestra; 0 = apagado.")
    parser.add_argument("--aug-gamma", type=float, default=0.0,
                        help="Amplitud log del aumento de gamma por muestra; 0 = apagado.")
    parser.add_argument("--backbone-init", default=None)
    args = parser.parse_args()
    initialization_options = [args.resume, args.backbone_init, args.week9_checkpoint]
    if sum(value is not None for value in initialization_options) > 1:
        parser.error("Usa solo una opción: --resume, --backbone-init o --week9-checkpoint.")
    if args.freeze_week9_epochs < 0:
        parser.error("--freeze-week9-epochs no puede ser negativo.")
    if args.freeze_week9_epochs > 0 and not args.week9_checkpoint:
        parser.error("--freeze-week9-epochs requiere --week9-checkpoint.")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience no puede ser negativo.")
    if args.early_stopping_min_delta < 0:
        parser.error("--early-stopping-min-delta no puede ser negativo.")
    if not 0 <= args.early_stopping_min_epochs <= args.epochs:
        parser.error("--early-stopping-min-epochs debe estar entre 0 y --epochs.")
    if args.stop_dice is not None and not 0 <= args.stop_dice <= 1:
        parser.error("--stop-dice debe estar entre 0 y 1.")
    if args.stop_instance_dice is not None and not 0 <= args.stop_instance_dice <= 1:
        parser.error("--stop-instance-dice debe estar entre 0 y 1.")
    if args.aug_mm and args.aug_geo <= 0:
        parser.error("--aug-mm requiere --aug-geo mayor que 0.")
    if args.normalizar and not args.canonical:
        parser.error("--normalizar requiere --canonical para fijar primero la orientación de cabecera.")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    started = time.time()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    root = Path(args.cache_root)
    cache_dir = root / cache_dir_name(args.size, args.mode)
    splits = read_split(root)
    print(f"Casos: train={len(splits['train'])} val={len(splits['val'])} | {cache_dir}", flush=True)
    train = load_slices(cache_dir, splits["train"], args.margin, canonical=args.canonical,
                        normalize_rotation=args.normalizar)
    val = load_slices(cache_dir, splits["val"], args.margin, canonical=args.canonical,
                      normalize_rotation=args.normalizar)
    # Un corte de cada val_stride por caso; los vecinos del contexto salen de todos los cortes.
    val_pick = np.concatenate([np.flatnonzero(val["case_lo"] == lo)[::args.val_stride]
                               for lo in np.unique(val["case_lo"])])
    print(f"Cortes train={len(train['images'])} (con hueso {int(train['has_bone'].sum())}) | "
          f"val={len(val_pick)} | carga {time.time() - started:.0f}s", flush=True)
    if args.normalizar:
        accepted_train = sum(bool(row.get("accepted")) for row in train["normalization"].values())
        accepted_val = sum(bool(row.get("accepted")) for row in val["normalization"].values())
        print(f"Enderezado pélvico aceptado: train={accepted_train}/{len(train['normalization'])} | "
              f"val={accepted_val}/{len(val['normalization'])}", flush=True)
    val_x, val_y = to_batch(gather_context(val, val_pick, args.context), val["labels"][val_pick])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cbam = args.cbam != "none"
    cbam_stages = int(args.cbam) if use_cbam else 1
    in_channels = 2 * args.context + 1
    model = PengwinMultiTaskModel(base_channels=args.base_channels, instance_mode=args.instance_mode,
                                  in_channels=in_channels, use_cbam=use_cbam, cbam_stages=cbam_stages,
                                  learnable_gamma=args.learnable_gamma,
                                  deep_supervision=args.deep_supervision_weight > 0).to(device)
    week9_transfer = None
    continued_from = None
    resume_checkpoint = None
    if args.week9_checkpoint:
        week9_transfer = load_week9_initialization(model, args.week9_checkpoint, device)
        if week9_transfer["canonical"] != bool(args.canonical):
            raise ValueError(
                f"Orientación incompatible: Semana 9 canonical={week9_transfer['canonical']} "
                f"y Semana 10 canonical={bool(args.canonical)}."
            )
        if week9_transfer["aug_mm"] != bool(args.aug_mm):
            raise ValueError(
                f"Aumento físico incompatible: Semana 9 aug_mm={week9_transfer['aug_mm']} "
                f"y Semana 10 aug_mm={bool(args.aug_mm)}."
            )
    if args.backbone_init:
        state = torch.load(args.backbone_init, map_location=device)
        state = state.get("backbone", state)
        current = model.backbone.state_dict()
        loadable = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
        model.backbone.load_state_dict(loadable, strict=False)
        print(f"Backbone preentrenado: {len(loadable)}/{len(current)} tensores cargados de {args.backbone_init}",
              flush=True)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint de Semana 10: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if int(resume_checkpoint.get("base_channels", -1)) != args.base_channels:
            raise ValueError("El ancho de --resume no coincide con --base-channels.")
        if int(resume_checkpoint.get("in_channels", -1)) != in_channels:
            raise ValueError("Los canales de --resume no coinciden con --context.")
        if bool(resume_checkpoint.get("canonical", False)) != bool(args.canonical):
            raise ValueError("La orientacion canonical de --resume no coincide con la corrida actual.")
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        week9_transfer = resume_checkpoint.get("week9_transfer")
        continued_from = {
            "path": str(resume_path),
            "best_validation": resume_checkpoint.get("best_validation"),
            "week9_transfer": week9_transfer,
        }
        print(f"Continuacion Semana 10 desde {resume_path}: todos los pesos cargados.", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {n_params:,} parámetros | CBAM={args.cbam} gamma aprendible={args.learnable_gamma} | {device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1),
                                                            eta_min=args.lr / 50) if args.cosine else None)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    weights = MultiTaskLossWeights(
        detection=args.detection_weight, classification=args.classification_weight,
        semantic=args.semantic_weight, instance=args.instance_weight,
        semantic_dice=args.semantic_dice_weight, boundary=args.boundary_weight,
        centers=args.center_weight, offsets=args.offset_weight,
        instance_dice=args.instance_dice_weight, fracture_weight=args.fracture_weight,
        secondary_weight=args.secondary_weight, focal_gamma=args.focal_gamma,
        tversky_beta=args.tversky_beta, deep_supervision=args.deep_supervision_weight)

    ema_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                 if args.ema > 0 else None)
    updates = 0

    def ema_update() -> None:
        decay = min(args.ema, (1 + updates) / (10 + updates))
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema_state[key].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                ema_state[key].copy_(value)

    @contextlib.contextmanager
    def averaged_weights():
        """Dentro del bloque el modelo usa los pesos promediados (si ``--ema``)."""
        if ema_state is None:
            yield
            return
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
        try:
            yield
        finally:
            model.load_state_dict(backup)

    best_score, best_state, best_metrics, history = -float("inf"), None, None, []
    last_metrics = None
    best_selection_value = -float("inf")
    early_best_value = -float("inf")
    early_bad_evaluations = 0
    stop_reason = None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint_dict(state: dict, metrics: dict | None) -> dict:
        return {"model": state, "base_channels": args.base_channels, "instance_mode": args.instance_mode,
                "in_channels": in_channels, "context": args.context,
                "deep_supervision": args.deep_supervision_weight > 0, "ema": args.ema,
                "disable_cudnn": args.disable_cudnn,
                "use_cbam": use_cbam, "cbam_stages": cbam_stages, "canonical": args.canonical,
                "normalizar": args.normalizar,
                "learnable_gamma": args.learnable_gamma, "augment": args.augment, "aug_geo": args.aug_geo, "aug_gamma": args.aug_gamma, "aug_rot_wide": args.aug_rot_wide, "aug_rot_deg": args.aug_rot_deg, "aug_lowres": args.aug_lowres, "cosine": args.cosine,
                "aug_mm": args.aug_mm,
                "input_size": args.size, "mode": args.mode, "seed": args.seed,
                "fracture_weight": args.fracture_weight, "secondary_weight": args.secondary_weight,
                "week9_transfer": week9_transfer, "continued_from": continued_from,
                "best_validation": metrics, "args": vars(args)}

    # El punto de partida también compite como mejor checkpoint: si el ajuste
    # fino empeora, se conservan exactamente los pesos recibidos de la época 25.
    if resume_checkpoint is not None and isinstance(resume_checkpoint.get("best_validation"), dict):
        best_metrics = resume_checkpoint["best_validation"]
        best_score = (best_metrics["segmentation_dice_macro"] + best_metrics["detection_map50"]
                      + best_metrics["instance_dice_matched"])
        baseline_values = {
            "score": best_score,
            "semantic_dice": float(best_metrics["segmentation_dice_macro"]),
            "instance_dice": float(best_metrics["instance_dice_matched"]),
        }
        best_selection_value = baseline_values[args.checkpoint_metric]
        early_best_value = baseline_values[args.early_stopping_metric]
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(checkpoint_dict(best_state, best_metrics), out_path)
        print(f"Referencia inicial conservada: {args.checkpoint_metric}={best_selection_value:.4f}",
              flush=True)

    def set_week9_trainable(trainable: bool) -> None:
        for module in (model.backbone, model.detection_head):
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)

    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        freeze_week9 = bool(args.week9_checkpoint and epoch <= args.freeze_week9_epochs)
        set_week9_trainable(not freeze_week9)
        model.train()
        if freeze_week9:
            # Congelar también estadísticas BatchNorm; así los módulos heredados
            # permanecen exactamente como salieron del notebook de Semana 9.
            model.backbone.eval()
            model.detection_head.eval()
        if epoch == 1 or epoch == args.freeze_week9_epochs + 1:
            print(f"Módulos heredados Semana 9: {'congelados' if freeze_week9 else 'ajustables'} "
                  f"(epoch {epoch})", flush=True)
        order = epoch_indices(train["has_bone"], rng, args.neg_fraction, args.per_epoch,
                              train["has_secondary"], args.secondary_oversample)
        losses = []
        t_epoch = time.time()
        for start in range(0, len(order), args.batch_size):
            pick = np.sort(order[start:start + args.batch_size])
            x, y = to_batch(gather_context(train, pick, args.context), train["labels"][pick])
            w = torch.ones(y.shape, dtype=torch.float32)
            if args.augment:
                flip = torch.rand(x.shape[0]) < 0.5
                if flip.any():
                    x[flip], y[flip] = flip_sagittal(x[flip], y[flip])
                gain = 1.0 + 0.1 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5)
                bias = 0.05 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5)
                x = (x * gain + bias).clamp(0.0, 1.0)
            if args.aug_geo > 0:
                x, y = random_affine(x, y, strength=args.aug_geo, wide_rotation=args.aug_rot_wide,
                                     rotation_degrees=args.aug_rot_deg,
                                     spacing_yx=train["spacing_yx"][pick] if args.aug_mm else None)
            if args.aug_lowres > 0:
                x = random_lowres(x, args.aug_lowres)
            if args.aug_gamma > 0:
                gamma = torch.exp((torch.rand(x.shape[0], 1, 1, 1) * 2 - 1) * args.aug_gamma)
                x = x.clamp(0.0, 1.0) ** gamma
            if args.fracture_weight > 0:
                w = fracture_distance_weight(y, args.fracture_weight)
            target = labels_to_grid(y)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss, loss_parts = multitask_loss(model(x.to(device, non_blocking=True)), target.to(device),
                                                  y.to(device), weights=weights, fracture=w.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Pérdida no finita en epoch={epoch}, batch={start // args.batch_size}: "
                    f"{loss_parts}"
                )
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if ema_state is not None:
                updates += 1
                ema_update()
            losses.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        over_budget = args.max_minutes is not None and (time.time() - started) / 60 > args.max_minutes
        early_stop_now = False
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs or over_budget:
            use_primary = args.instance_mode == "primary"
            with averaged_weights():
                metrics = evaluate(model, val_x, val_y, device, args.batch_size * 2, args.confidence,
                                   two_pass=False)
                eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            last_metrics = metrics
            score = (metrics["segmentation_dice_macro"] + metrics["detection_map50"]
                     + metrics["instance_dice_matched"])
            semantic_dice = float(metrics["segmentation_dice_macro"])
            instance_dice = float(metrics["instance_dice_matched"])
            monitored_values = {
                "semantic_dice": semantic_dice,
                "instance_dice": instance_dice,
                "score": score,
            }
            monitored_value = monitored_values[args.early_stopping_metric]
            selection_value = monitored_values[args.checkpoint_metric]
            if monitored_value > early_best_value + args.early_stopping_min_delta:
                early_best_value = monitored_value
                early_bad_evaluations = 0
            else:
                early_bad_evaluations += 1

            has_targets = args.stop_dice is not None or args.stop_instance_dice is not None
            semantic_target_reached = args.stop_dice is None or semantic_dice >= args.stop_dice
            instance_target_reached = (args.stop_instance_dice is None or
                                       instance_dice >= args.stop_instance_dice)
            targets_reached = has_targets and semantic_target_reached and instance_target_reached
            if epoch >= args.early_stopping_min_epochs:
                if targets_reached:
                    stop_reason = (f"objetivos Dice alcanzados: semantico={semantic_dice:.4f}, "
                                   f"instancias={instance_dice:.4f}")
                    early_stop_now = True
                elif (args.early_stopping_patience > 0 and
                      early_bad_evaluations >= args.early_stopping_patience):
                    stop_reason = (f"{args.early_stopping_metric} sin mejorar al menos "
                                   f"{args.early_stopping_min_delta:.4f} "
                                   f"durante {early_bad_evaluations} evaluaciones")
                    early_stop_now = True
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "score": score,
                            "early_stopping_bad_evaluations": early_bad_evaluations,
                            "early_stopping_metric": args.early_stopping_metric,
                            "semantic_dice_target_reached": semantic_target_reached,
                            "instance_dice_target_reached": instance_target_reached,
                            "all_dice_targets_reached": targets_reached, **metrics})

            # Un checkpoint que cumple todos los objetivos Dice tiene prioridad;
            # dentro del mismo estado se usa la métrica de selección configurada.
            best_semantic_reached = (best_metrics is not None and
                                     (args.stop_dice is None or
                                      best_metrics["segmentation_dice_macro"] >= args.stop_dice))
            best_instance_reached = (best_metrics is not None and
                                     (args.stop_instance_dice is None or
                                      best_metrics["instance_dice_matched"] >= args.stop_instance_dice))
            best_reached = has_targets and best_semantic_reached and best_instance_reached
            improves_checkpoint = ((targets_reached and not best_reached) or
                                   (targets_reached == best_reached and
                                    selection_value > best_selection_value))
            if improves_checkpoint:
                best_score, best_metrics = score, metrics
                best_selection_value = selection_value
                best_state = eval_state
                torch.save(checkpoint_dict(best_state, best_metrics), out_path)
            print(f"epoch {epoch:03d} loss={np.mean(losses):.4f} dice={metrics['segmentation_dice_macro']:.4f} "
                  f"map50={metrics['detection_map50']:.4f} inst={metrics['instance_dice_matched']:.4f} "
                  f"({time.time() - t_epoch:.0f}s/época, total {(time.time() - started) / 60:.1f} min)", flush=True)
        else:
            print(f"epoch {epoch:03d} loss={np.mean(losses):.4f} ({time.time() - t_epoch:.0f}s)", flush=True)
        if early_stop_now:
            print(f"Parada temprana en epoch {epoch}: {stop_reason}. Se conserva el mejor checkpoint.",
                  flush=True)
            stopped_early = True
            break
        if over_budget:
            print(f"Límite de {args.max_minutes} min alcanzado; se guarda el mejor punto.", flush=True)
            stopped_early = True
            stop_reason = f"limite de tiempo de {args.max_minutes} minutos"
            break

    last_path = None
    if args.last_output:
        last_path = Path(args.last_output)
        if last_path.resolve() == out_path.resolve():
            raise ValueError("--last-output debe ser distinto de --output.")
        last_path.parent.mkdir(parents=True, exist_ok=True)
        with averaged_weights():
            last_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        last_checkpoint = checkpoint_dict(last_state, last_metrics)
        last_checkpoint["checkpoint_kind"] = "ultima_epoca"
        last_checkpoint["epoch_completed"] = epoch
        torch.save(last_checkpoint, last_path)
        print(f"Checkpoint de última época: {last_path}", flush=True)

    metrics_path = Path(args.metrics_output) if args.metrics_output else out_path.with_suffix(".json")
    metrics_path.write_text(json.dumps({"best": best_metrics, "history": history,
                                        "stopped_early": stopped_early, "stop_reason": stop_reason,
                                        "epochs_completed": epoch, "args": vars(args),
                                        "best_checkpoint": str(out_path),
                                        "last_checkpoint": str(last_path) if last_path else None,
                                        "minutes": (time.time() - started) / 60}, indent=2))
    print(f"Checkpoint: {out_path} | métricas: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
