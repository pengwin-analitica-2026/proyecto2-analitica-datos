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
- ``--secondary-oversample``: muestrea más los cortes con fragmentos secundarios;
- ``--backbone-init``: pesos del backbone preentrenados con ``pretrain_backbone.py``;
- ``--aug-mm``: el aumento geométrico gira en milímetros reales (``aumento_fisico.py``), no en
  la grilla de píxeles estirada; sin la opción se conserva el comportamiento anterior;
- ``--aug-noise``: ruido gaussiano en el 30 % de las muestras, con sigma al azar hasta ese valor
  (escala 0-1); 0 = apagado.

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

import math

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aumento_fisico import mm_de_cortes, mm_por_pixel, random_affine_mm
from build_case_cache import cache_dir_name, from_uint8, load_case_cache
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
                keep_negatives: bool = True, canonical: bool = False) -> dict[str, np.ndarray]:
    """Carga los cortes de interés de cada caso: extensión ósea ± ``margin``.

    Devuelve arrays uint8 concatenados y un vector ``has_bone`` para muestrear.
    """
    images, labels, bone, case_of, lows, highs, secondary = [], [], [], [], [], [], []
    mm_x, mm_y = [], []
    total = 0
    for case_id in case_ids:
        data = load_case_cache(cache_dir, case_id, canonical=canonical)
        has_bone = (data["labels"] != 0).any(axis=(1, 2))
        where = np.flatnonzero(has_bone)
        if len(where) == 0:
            continue
        low, high = max(int(where[0]) - margin, 0), min(int(where[-1]) + margin + 1, len(has_bone))
        picks = np.arange(low, high, stride)
        if not keep_negatives:
            picks = picks[has_bone[picks]]
        images.append(data["images"][picks]); labels.append(data["labels"][picks])
        bone.append(has_bone[picks]); case_of.extend([case_id] * len(picks))
        lab = data["labels"][picks]
        secondary.append(((lab % 10 != 1) & (lab > 0)).any(axis=(1, 2)))  # IDs 2..10, 12..20, 22..30
        lows.append(np.full(len(picks), total)); highs.append(np.full(len(picks), total + len(picks) - 1))
        mx, my = mm_por_pixel(data)                     # mm por píxel del caché, de la cabecera
        mm_x.append(np.full(len(picks), mx, dtype=np.float32))
        mm_y.append(np.full(len(picks), my, dtype=np.float32))
        total += len(picks)
    return {"images": np.concatenate(images), "labels": np.concatenate(labels),
            "has_bone": np.concatenate(bone), "case": np.asarray(case_of),
            "has_secondary": np.concatenate(secondary),
            "case_lo": np.concatenate(lows), "case_hi": np.concatenate(highs),
            "mm_x": np.concatenate(mm_x), "mm_y": np.concatenate(mm_y)}


def to_batch(images_u8: np.ndarray, labels_u8: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Imágenes [N,H,W] (un canal) o [N,C,H,W] (con contexto) → tensores float/int64."""
    x = torch.from_numpy(from_uint8(images_u8))
    if x.ndim == 3:
        x = x.unsqueeze(1)
    y = torch.from_numpy(labels_u8.astype(np.int64))
    return x, y


def random_affine(x: torch.Tensor, y: torch.Tensor, strength: float = 1.0,
                  probability: float = 0.8, wide_rotation: float = 0.0,
                  rotation_degrees: float = 12.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Aumento geométrico por muestra: escala, proporción, rotación y traslación.

    Los volúmenes de PENGWIN llegan con campos de visión, espaciados y proporciones distintas
    (36 de 100 casos no son 512x512), y el caché los estira a un cuadrado. Con 70 pacientes de
    entrenamiento la red memoriza esa geometría: la pérdida de entrenamiento sigue bajando
    mientras la validación se estanca. Este aumento expone a la red a esa variación.

    ``strength`` = 1: escala 0.80-1.25, proporción 0.85-1.18, rotación +-``rotation_degrees``
    (12 por defecto), traslación +-8 % del lado. ``wide_rotation`` es la fracción de muestras con rotación libre (+-180 grados). La imagen se remuestrea bilineal y la etiqueta con vecino más cercano, así
    los IDs de fragmento no se mezclan. Las cajas de detección se derivan después de la etiqueta.
    """
    n = x.shape[0]
    scale = torch.exp((torch.rand(n) * 2 - 1) * 0.22 * strength)
    aspect = torch.exp((torch.rand(n) * 2 - 1) * 0.16 * strength)
    angle = (torch.rand(n) * 2 - 1) * math.radians(rotation_degrees) * strength
    shift = (torch.rand(n, 2) * 2 - 1) * 0.16 * strength
    if wide_rotation > 0:
        # Pacientes girados en el escáner (decúbito lateral, prono): cualquier ángulo. Una rotación
        # conserva la lateralidad, así que los IDs de coxal izquierdo y derecho no cambian.
        wide = torch.rand(n) < wide_rotation
        angle[wide] = (torch.rand(int(wide.sum())) * 2 - 1) * math.pi
    keep = torch.rand(n) >= probability            # muestras que quedan sin tocar
    scale[keep], aspect[keep], angle[keep], shift[keep] = 1.0, 1.0, 0.0, 0.0
    sx, sy = scale * aspect, scale / aspect
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos / sx, -sin / sy, shift[:, 0]
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin / sx, cos / sy, shift[:, 1]
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x_out = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y_out = torch.nn.functional.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                                            align_corners=False)[:, 0].long()
    return x_out, y_out


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
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--output", default="outputs/s10/modelo.pth")
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
    parser.add_argument("--aug-geo", type=float, default=0.0,
                        help="Fuerza del aumento geométrico (escala, proporción, rotación, traslación); 0 = apagado.")
    parser.add_argument("--aug-rot-deg", type=float, default=12.0,
                        help="Amplitud de la rotación del aumento geométrico, en grados (con --aug-geo).")
    parser.add_argument("--aug-rot-wide", type=float, default=0.0,
                        help="Fracción de muestras con rotación libre de +-180 grados (requiere --aug-geo).")
    parser.add_argument("--aug-lowres", type=float, default=0.0,
                        help="Probabilidad de simular baja resolución por muestra; 0 = apagado.")
    parser.add_argument("--aug-gamma", type=float, default=0.0,
                        help="Amplitud log del aumento de gamma por muestra; 0 = apagado.")
    parser.add_argument("--aug-noise", type=float, default=0.0,
                        help="Sigma máxima del ruido gaussiano (escala 0-1) en el 30 %% de las muestras; 0 = apagado.")
    parser.add_argument("--aug-mm", action="store_true",
                        help="Aumento geométrico en milímetros reales (aumento_fisico.py), no en píxeles.")
    parser.add_argument("--backbone-init", default=None)
    args = parser.parse_args()

    started = time.time()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    root = Path(args.cache_root)
    cache_dir = root / cache_dir_name(args.size, args.mode)
    splits = read_split(root)
    print(f"Casos: train={len(splits['train'])} val={len(splits['val'])} | {cache_dir}", flush=True)
    train = load_slices(cache_dir, splits["train"], args.margin, canonical=args.canonical)
    val = load_slices(cache_dir, splits["val"], args.margin, canonical=args.canonical)
    # Un corte de cada val_stride por caso; los vecinos del contexto salen de todos los cortes.
    val_pick = np.concatenate([np.flatnonzero(val["case_lo"] == lo)[::args.val_stride]
                               for lo in np.unique(val["case_lo"])])
    print(f"Cortes train={len(train['images'])} (con hueso {int(train['has_bone'].sum())}) | "
          f"val={len(val_pick)} | carga {time.time() - started:.0f}s", flush=True)
    val_x, val_y = to_batch(gather_context(val, val_pick, args.context), val["labels"][val_pick])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cbam = args.cbam != "none"
    cbam_stages = int(args.cbam) if use_cbam else 1
    in_channels = 2 * args.context + 1
    model = PengwinMultiTaskModel(base_channels=args.base_channels, instance_mode=args.instance_mode,
                                  in_channels=in_channels, use_cbam=use_cbam, cbam_stages=cbam_stages,
                                  learnable_gamma=args.learnable_gamma,
                                  deep_supervision=args.deep_supervision_weight > 0).to(device)
    if args.backbone_init:
        state = torch.load(args.backbone_init, map_location=device)
        state = state.get("backbone", state)
        current = model.backbone.state_dict()
        loadable = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
        model.backbone.load_state_dict(loadable, strict=False)
        print(f"Backbone preentrenado: {len(loadable)}/{len(current)} tensores cargados de {args.backbone_init}",
              flush=True)
    if args.resume:
        state = dict(torch.load(args.resume, map_location=device)["model"])
        current = model.state_dict()
        for key in [k for k, v in state.items() if k in current and tuple(v.shape) != tuple(current[k].shape)]:
            state.pop(key)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Reanudado de {args.resume}: capas nuevas={len(missing)} sobrantes={len(unexpected)}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {n_params:,} parámetros | CBAM={args.cbam} γ aprendible={args.learnable_gamma} | {device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1),
                                                            eta_min=args.lr / 50) if args.cosine else None)
    amp = device.type == "cuda"
    # torch.amp.GradScaler existe desde torch 2.3; antes es torch.cuda.amp.GradScaler (mismo comportamiento).
    # Así el entrenamiento (y sus pruebas) corre también con torch 2.2, la versión del Mac x86.
    scaler = (torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp))
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
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint_dict(state: dict, metrics: dict | None) -> dict:
        return {"model": state, "base_channels": args.base_channels, "instance_mode": args.instance_mode,
                "in_channels": in_channels, "context": args.context,
                "deep_supervision": args.deep_supervision_weight > 0, "ema": args.ema,
                "use_cbam": use_cbam, "cbam_stages": cbam_stages, "canonical": args.canonical,
                "learnable_gamma": args.learnable_gamma, "augment": args.augment, "aug_geo": args.aug_geo, "aug_gamma": args.aug_gamma, "aug_rot_wide": args.aug_rot_wide, "aug_rot_deg": args.aug_rot_deg, "aug_lowres": args.aug_lowres, "aug_mm": args.aug_mm, "aug_noise": args.aug_noise, "cosine": args.cosine,
                "input_size": args.size, "mode": args.mode, "seed": args.seed,
                "fracture_weight": args.fracture_weight, "secondary_weight": args.secondary_weight,
                "best_validation": metrics, "args": vars(args)}

    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
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
            if args.aug_geo > 0 and args.aug_mm:
                x, y = random_affine_mm(x, y, mm_de_cortes(train, pick), strength=args.aug_geo,
                                        wide_rotation=args.aug_rot_wide,
                                        rotation_degrees=args.aug_rot_deg)
            elif args.aug_geo > 0:
                x, y = random_affine(x, y, strength=args.aug_geo, wide_rotation=args.aug_rot_wide,
                                     rotation_degrees=args.aug_rot_deg)
            if args.aug_lowres > 0:
                x = random_lowres(x, args.aug_lowres)
            if args.aug_gamma > 0:
                gamma = torch.exp((torch.rand(x.shape[0], 1, 1, 1) * 2 - 1) * args.aug_gamma)
                x = x.clamp(0.0, 1.0) ** gamma
            if args.aug_noise > 0:
                on = torch.rand(x.shape[0]) < 0.3
                sigma = torch.rand(x.shape[0], 1, 1, 1) * args.aug_noise
                x[on] = (x[on] + torch.randn_like(x[on]) * sigma[on]).clamp(0.0, 1.0)
            if args.fracture_weight > 0:
                w = fracture_distance_weight(y, args.fracture_weight)
            target = labels_to_grid(y)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss, _ = multitask_loss(model(x.to(device, non_blocking=True)), target.to(device),
                                         y.to(device), weights=weights, fracture=w.to(device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if ema_state is not None:
                updates += 1
                ema_update()
            losses.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        over_budget = args.max_minutes is not None and (time.time() - started) / 60 > args.max_minutes
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs or over_budget:
            use_primary = args.instance_mode == "primary"
            with averaged_weights():
                metrics = evaluate(model, val_x, val_y, device, args.batch_size * 2, args.confidence,
                                   two_pass=False)
                eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            score = (metrics["segmentation_dice_macro"] + metrics["detection_map50"]
                     + metrics["instance_dice_matched"])
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "score": score, **metrics})
            if score > best_score:
                best_score, best_metrics = score, metrics
                best_state = eval_state
                torch.save(checkpoint_dict(best_state, best_metrics), out_path)
            print(f"epoch {epoch:03d} loss={np.mean(losses):.4f} dice={metrics['segmentation_dice_macro']:.4f} "
                  f"map50={metrics['detection_map50']:.4f} inst={metrics['instance_dice_matched']:.4f} "
                  f"({time.time() - t_epoch:.0f}s/época, total {(time.time() - started) / 60:.1f} min)", flush=True)
        else:
            print(f"epoch {epoch:03d} loss={np.mean(losses):.4f} ({time.time() - t_epoch:.0f}s)", flush=True)
        if over_budget:
            print(f"Límite de {args.max_minutes} min alcanzado; se guarda el mejor punto.", flush=True)
            stopped_early = True
            break

    metrics_path = Path(args.metrics_output) if args.metrics_output else out_path.with_suffix(".json")
    metrics_path.write_text(json.dumps({"best": best_metrics, "history": history,
                                        "stopped_early": stopped_early, "args": vars(args),
                                        "minutes": (time.time() - started) / 60}, indent=2))
    print(f"Checkpoint: {out_path} | métricas: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
