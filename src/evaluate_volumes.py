"""Evaluación sobre el volumen completo de cada caso (regla de medición de la Semana 10).

Diferencias con ``evaluate()`` de ``train_week10_multitask.py``:

- recorre **todos los cortes** de cada caso (caché por caso de
  ``build_case_cache.py``), no una muestra del centro del hueso;
- reporta la detección de dos formas: la cabeza grid **cruda** (con NMS propio)
  y la **refinada** con la máscara semántica, que es lo que se medía antes;
- separa el Dice por fragmento en principal y secundario;
- guarda un registro por corte, así cualquier subconjunto (solo cortes con
  hueso, centro, extremos) se resume después sin volver a inferir.

Las métricas se calculan con las mismas funciones de ``pengwin_metrics``; una
prueba comprueba que, sobre los mismos cortes, coinciden con ``evaluate()``.

Uso:

    python src/evaluate_volumes.py --checkpoint outputs/gamma/w10g_384neg.pth \\
        --cache-dir data/cache/s10/cache_384 --split val --tta \\
        --output outputs/s10/linea_base_val.json

Con ``--normalizar`` cada volumen pasa por el paso 2 de ``normalizacion.py`` (giro axial del
paciente, estimado con la simetría izquierda-derecha de la pelvis en milímetros) antes de entrar
al modelo; el paso 1 (orientación con los metadatos) ya lo hace el caché canónico. Se aplica
después de ``--stress``, así ``--stress rot=60 --normalizar`` mide si el giro se recupera. Imagen y
etiqueta reciben el mismo giro, de modo que las métricas se comparan en el marco enderezado.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aumento_fisico import mm_por_pixel
from build_case_cache import from_uint8, load_case_cache
from instance_postprocess import primary_secondary_to_instances, semantic_to_instances
from pengwin_metrics import detection_metrics, multilabel_f1_auc, refine_boxes_with_semantic
from week9_detection import boxes_from_label, decode_predictions

REGIONS = ("sacro", "coxal_izquierdo", "coxal_derecho")
REGION_RANGES = ((1, 10), (11, 20), (21, 30))
PRIMARY_IDS = (1, 11, 21)

OBJETIVOS = {
    "classification_f1_macro": 0.85, "classification_auc_macro": 0.85,
    "detection_iou_mean": 0.65, "detection_map50": 0.65, "detection_map50_95": 0.40,
    "segmentation_dice_macro": 0.85, "segmentation_iou_macro": 0.70,
    "instance_dice_matched": 0.85, "instance_iou_matched": 0.70,
}


@dataclass
class SliceRecord:
    """Todo lo necesario para resumir un corte sin volver a inferir."""
    case_id: str
    z: int
    has_bone: bool
    rel_pos: float                      # posición en la extensión ósea: 0 = primer corte con hueso, 1 = último
    cls_logits: np.ndarray              # (3,)
    cls_target: np.ndarray              # (3,)
    gt_boxes: list
    det_raw: list                       # todas las cajas tras NMS
    det_raw_one: list                   # una caja por región (la de mayor puntaje)
    det_refined: list                   # una por región, límites tomados de la semántica
    sem_inter: np.ndarray               # (3,) píxeles por región
    sem_pred: np.ndarray
    sem_gt: np.ndarray
    fragments: list = field(default_factory=list)   # (gt_id, dice, iou, area_gt)


def semantic_from_labels(labels: np.ndarray) -> np.ndarray:
    semantic = np.zeros(labels.shape, dtype=np.uint8)
    for index, (low, high) in enumerate(REGION_RANGES):
        semantic[(labels >= low) & (labels <= high)] = index + 1
    return semantic


def matched_fragments(prediction: np.ndarray, truth: np.ndarray) -> list[tuple[int, float, float, int]]:
    """Empareja fragmentos de un corte, igual que ``matched_instance_metrics``.

    Devuelve una tupla (id real, Dice, IoU, área real) por fragmento real. Los
    fragmentos sin pareja reciben cero.
    """
    rows = []
    for low, high in REGION_RANGES:
        gt_ids = [int(v) for v in np.unique(truth) if low <= v <= high]
        pred_ids = [int(v) for v in np.unique(prediction) if low <= v <= high]
        used: set[int] = set()
        for gt_id in gt_ids:
            gt = truth == gt_id
            area = int(gt.sum())
            best_id, best_iou = None, 0.0
            for pred_id in pred_ids:
                if pred_id in used:
                    continue
                candidate = prediction == pred_id
                iou = np.logical_and(gt, candidate).sum() / max(np.logical_or(gt, candidate).sum(), 1)
                if iou > best_iou:
                    best_id, best_iou = pred_id, iou
            if best_id is None:
                rows.append((gt_id, 0.0, 0.0, area))
                continue
            used.add(best_id)
            candidate = prediction == best_id
            dice = 2 * np.logical_and(gt, candidate).sum() / max(gt.sum() + candidate.sum(), 1)
            rows.append((gt_id, float(dice), float(best_iou), area))
    return rows


def relative_positions(has_bone: np.ndarray) -> np.ndarray:
    """Posición de cada corte respecto a la extensión ósea del caso."""
    bone_z = np.flatnonzero(has_bone)
    if len(bone_z) < 2:
        return np.full(len(has_bone), np.nan)
    first, last = bone_z[0], bone_z[-1]
    return (np.arange(len(has_bone)) - first) / (last - first)


@torch.inference_mode()
def evaluate_case(model: torch.nn.Module, case_id: str, images_u8: np.ndarray, labels: np.ndarray,
                  device: torch.device, batch_size: int = 16, confidence: float = 0.10,
                  two_pass: bool = False, keep_predictions: bool = False
                  ) -> tuple[list[SliceRecord], dict | None]:
    """Infiere un caso completo y devuelve un registro por corte."""
    model.eval()
    depth = len(images_u8)
    has_bone = (labels != 0).any(axis=(1, 2))
    rel = relative_positions(has_bone)
    records: list[SliceRecord] = []
    kept_semantic, kept_instance, kept_ps = [], [], []
    context = (int(getattr(model, "in_channels", 1)) - 1) // 2
    for start in range(0, depth, batch_size):
        chunk = slice(start, min(start + batch_size, depth))
        if context > 0:
            # 2.5D: los vecinos z±d como canales; en los extremos del volumen se repite el borde.
            z = np.arange(chunk.start, chunk.stop)
            volume = np.stack([images_u8[np.clip(z + d, 0, depth - 1)]
                               for d in range(-context, context + 1)], axis=1)
            x = torch.from_numpy(from_uint8(volume)).to(device)
        else:
            x = torch.from_numpy(from_uint8(images_u8[chunk]))[:, None].to(device)
        y_np = labels[chunk]
        y = torch.from_numpy(y_np.astype(np.int64))
        output = model(x)

        semantic = output["semantic"].argmax(1).cpu().numpy().astype(np.uint8)
        instance_mode = getattr(model, "instance_mode", "ids")
        if instance_mode == "primary":
            if two_pass:
                from week10_multitask import two_pass_instances
                instance_argmax = two_pass_instances(model, x, output["semantic"].argmax(1)).cpu().numpy()
            else:
                instance_argmax = output["instance"].argmax(1).cpu().numpy()
            instances = primary_secondary_to_instances(instance_argmax, semantic)
            kept_ps.append(instance_argmax.astype(np.uint8))
        else:
            instances = semantic_to_instances(semantic, None)

        if instances.ndim == 2:   # lote de 1 slice: el post-proceso devuelve 2D
            instances = instances[None]
        raw = decode_predictions(output["detection"], confidence=confidence)
        raw_one = decode_predictions(output["detection"], confidence=confidence, max_por_clase=1)
        refined = refine_boxes_with_semantic(raw, semantic)
        truths = boxes_from_label(y)
        logits = output["classification"].cpu().numpy()
        gt_semantic = semantic_from_labels(y_np)

        for i in range(len(y_np)):
            inter = np.array([np.logical_and(semantic[i] == c, gt_semantic[i] == c).sum() for c in (1, 2, 3)])
            pred = np.array([(semantic[i] == c).sum() for c in (1, 2, 3)])
            gt = np.array([(gt_semantic[i] == c).sum() for c in (1, 2, 3)])
            z = start + i
            records.append(SliceRecord(
                case_id=case_id, z=z, has_bone=bool(has_bone[z]), rel_pos=float(rel[z]),
                cls_logits=logits[i], cls_target=(gt > 0).astype(np.float32),
                gt_boxes=truths[i], det_raw=raw[i], det_raw_one=raw_one[i], det_refined=refined[i],
                sem_inter=inter, sem_pred=pred, sem_gt=gt,
                fragments=matched_fragments(instances[i], y_np[i]),
            ))
        if keep_predictions:
            kept_semantic.append(semantic)
            kept_instance.append(instances.astype(np.uint8))
    predictions = None
    if keep_predictions:
        predictions = {"semantic": np.concatenate(kept_semantic), "instance": np.concatenate(kept_instance)}
        if kept_ps:   # mapa primario/secundario de 7 clases, entrada del post-proceso 3D
            predictions["primary_secondary"] = np.concatenate(kept_ps)
    return records, predictions


SUBSETS = {
    "todo": lambda r: True,
    "con_hueso": lambda r: r.has_bone,
    "centro_15_85": lambda r: r.has_bone and 0.15 <= r.rel_pos <= 0.85,
    "extremos": lambda r: r.has_bone and not (0.15 <= r.rel_pos <= 0.85),
    "sin_hueso": lambda r: not r.has_bone,
}


def summarize(records: list[SliceRecord], subset: str = "todo") -> dict:
    """Métricas de la sección 5 del enunciado sobre un subconjunto de cortes."""
    chosen = [r for r in records if SUBSETS[subset](r)]
    result: dict = {"n_slices": len(chosen), "n_cases": len({r.case_id for r in chosen})}
    if not chosen:
        return result
    result.update(multilabel_f1_auc(np.stack([r.cls_logits for r in chosen]),
                                    np.stack([r.cls_target for r in chosen])))
    truths = [r.gt_boxes for r in chosen]
    for name, attribute in (("", "det_refined"), ("_raw", "det_raw"), ("_raw_one", "det_raw_one")):
        metrics = detection_metrics([getattr(r, attribute) for r in chosen], truths)
        for key, value in metrics.items():
            result[key.replace("detection_", f"detection{name}_")] = value
    result["n_gt_boxes"] = int(sum(len(t) for t in truths))
    result["n_raw_boxes"] = int(sum(len(r.det_raw) for r in chosen))

    inter = np.sum([r.sem_inter for r in chosen], axis=0)
    pred = np.sum([r.sem_pred for r in chosen], axis=0)
    gt = np.sum([r.sem_gt for r in chosen], axis=0)
    present = gt > 0
    if present.any():
        dice = 2 * inter[present] / np.maximum(pred[present] + gt[present], 1)
        iou = inter[present] / np.maximum(pred[present] + gt[present] - inter[present], 1)
        result["segmentation_dice_macro"] = float(dice.mean())
        result["segmentation_iou_macro"] = float(iou.mean())
        for index in np.flatnonzero(present):
            result[f"segmentation_dice_{REGIONS[index]}"] = float(
                2 * inter[index] / max(pred[index] + gt[index], 1))
    else:
        result["segmentation_dice_macro"] = result["segmentation_iou_macro"] = 0.0

    fragments = [f for r in chosen for f in r.fragments]
    result["n_fragments"] = len(fragments)
    if fragments:
        result["instance_dice_matched"] = float(np.mean([f[1] for f in fragments]))
        result["instance_iou_matched"] = float(np.mean([f[2] for f in fragments]))
        for label, keep in (("primary", lambda f: f[0] in PRIMARY_IDS),
                            ("secondary", lambda f: f[0] not in PRIMARY_IDS)):
            group = [f for f in fragments if keep(f)]
            result[f"n_fragments_{label}"] = len(group)
            if group:
                result[f"instance_dice_{label}"] = float(np.mean([f[1] for f in group]))
                result[f"instance_iou_{label}"] = float(np.mean([f[2] for f in group]))
                result[f"instance_missed_{label}"] = float(np.mean([f[1] == 0.0 for f in group]))
    else:
        result["instance_dice_matched"] = result["instance_iou_matched"] = 0.0
    return result


def per_case_table(records: list[SliceRecord]) -> list[dict]:
    rows = []
    for case_id in sorted({r.case_id for r in records}):
        metrics = summarize([r for r in records if r.case_id == case_id], "todo")
        rows.append({"case_id": case_id, **{k: metrics.get(k) for k in (
            "n_slices", "n_fragments", "classification_f1_macro", "detection_raw_one_map50",
            "detection_map50", "segmentation_dice_macro", "instance_dice_matched",
            "instance_dice_primary", "instance_dice_secondary")}})
    return rows


def load_model(checkpoints: list[str], device: torch.device, tta: bool) -> tuple[torch.nn.Module, bool]:
    """Carga checkpoints de la Semana 10 anterior (``week10_multitask``)."""
    from evaluate_week10 import EnsembleTTA
    from week10_multitask import PengwinMultiTaskModel
    models, two_pass, canonical = [], False, []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu")
        canonical.append(bool(checkpoint.get("canonical", False)))
        model = PengwinMultiTaskModel(base_channels=int(checkpoint["base_channels"]),
                                      instance_mode=str(checkpoint.get("instance_mode", "ids")),
                                      in_channels=int(checkpoint.get("in_channels", 1)),
                                      use_cbam=bool(checkpoint.get("use_cbam", True)),
                                      cbam_stages=int(checkpoint.get("cbam_stages", 1)),
                                      learnable_gamma=bool(checkpoint.get("learnable_gamma", False)),
                                      deep_supervision=bool(checkpoint.get("deep_supervision", False)))
        model.load_state_dict(checkpoint["model"])
        models.append(model.to(device).eval())
        two_pass = two_pass or bool(checkpoint.get("two_pass", False))
    if len(set(canonical)) > 1:
        raise ValueError("No se pueden mezclar checkpoints entrenados con y sin orientación canónica")
    model = EnsembleTTA(models, tta).to(device) if len(models) > 1 or tta else models[0]
    model.canonical = canonical[0]     # el evaluador orienta los casos igual que en el entrenamiento
    return model, (False if len(models) > 1 or tta else two_pass)


def apply_stress(images_u8: np.ndarray, labels: np.ndarray, spec: str, mm_xy=None) -> tuple[np.ndarray, np.ndarray]:
    """Deforma un volumen de validación para medir robustez sin tocar test.

    ``spec`` combina claves separadas por coma: ``rot=60`` (grados), ``scale=0.7``,
    ``aspect=1.6`` (estira el eje vertical y encoge el horizontal), ``shift=0.15`` (fracción del
    lado, hacia la derecha) y ``lowres=2.5`` (reduce filas por ese factor y vuelve a ampliar).
    Imagen bilineal, etiqueta con vecino más cercano.
    """
    import math
    opts = {k: float(v) for k, v in (item.split("=") for item in spec.split(",") if item)}
    angle = math.radians(opts.get("rot", 0.0))
    scale, aspect, shift = opts.get("scale", 1.0), opts.get("aspect", 1.0), opts.get("shift", 0.0)
    sx, sy = scale / aspect, scale * aspect
    theta = torch.tensor([[math.cos(angle) / sx, -math.sin(angle) / sy, -2 * shift],
                          [math.sin(angle) / sx, math.cos(angle) / sy, 0.0]], dtype=torch.float32)
    if "rotmm" in opts:   # giro en milímetros reales (no deforma con píxeles no cuadrados)
        from aumento_fisico import theta_mm
        mm = torch.tensor([list(mm_xy) if mm_xy is not None else [1.0, 1.0]], dtype=torch.float32)
        theta = theta_mm(torch.tensor([math.radians(opts["rotmm"])]), torch.tensor([scale]),
                         torch.tensor([1.0 / aspect]), torch.tensor([[-2 * shift, 0.0]]), mm,
                         images_u8.shape[-2], images_u8.shape[-1])[0]
    out_i, out_l = [], []
    for start in range(0, len(images_u8), 64):
        x = torch.from_numpy(images_u8[start:start + 64].astype(np.float32))[:, None]
        y = torch.from_numpy(labels[start:start + 64].astype(np.float32))[:, None]
        grid = F.affine_grid(theta[None].expand(len(x), 2, 3), list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        y = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        if opts.get("lowres", 1.0) > 1.0:
            h, w = x.shape[-2:]
            x = F.interpolate(F.interpolate(x, size=(int(h / opts["lowres"]), int(w / (1 + (opts["lowres"] - 1) / 3))),
                                            mode="bilinear", align_corners=False),
                              size=(h, w), mode="bilinear", align_corners=False)
        out_i.append(x[:, 0].round().clamp(0, 255).numpy().astype(np.uint8))
        out_l.append(y[:, 0].numpy().astype(np.uint8))
    return np.concatenate(out_i), np.concatenate(out_l)


def cases_of_split(cache_dir: Path, split: str) -> list[str]:
    index = cache_dir.parent / "index.csv"
    with open(index, newline="") as stream:
        return [row["case_id"] for row in csv.DictReader(stream) if row["split"] == split]


def print_table(summary: dict) -> None:
    for key, goal in OBJETIVOS.items():
        value = summary.get(key)
        if value is None:
            continue
        mark = "cumple" if value >= goal else f"NO cumple (objetivo {goal})"
        print(f"  {key:34s} {value:.3f}  {mark}")
    for key in ("detection_raw_iou_mean", "detection_raw_map50", "detection_raw_map50_95",
                "detection_raw_one_iou_mean", "detection_raw_one_map50", "detection_raw_one_map50_95",
                "instance_dice_primary", "instance_dice_secondary", "instance_missed_secondary"):
        if key in summary:
            print(f"  {key:34s} {summary[key]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, nargs="+")
    parser.add_argument("--cache-dir", required=True, help="Carpeta cache_<size> de build_case_cache.py")
    parser.add_argument("--split", default="val")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--two-pass", action="store_true",
                        help="Segundo pase por región recortada (solo inferencia, modelo sin TTA).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-records", default=None, help="Guarda los registros por corte (pickle).")
    parser.add_argument("--stress", default=None,
                        help="Deformación de robustez sobre validación, p. ej. rot=60 o aspect=1.6,lowres=2.5.")
    parser.add_argument("--save-predictions", default=None,
                        help="Carpeta donde guardar semántica e instancias predichas por caso.")
    parser.add_argument("--normalizar", action="store_true",
                        help="Endereza el giro axial de cada volumen antes del modelo (normalizacion.py).")
    args = parser.parse_args()

    if args.split == "test":
        print("AVISO: test se evalúa una sola vez, con la configuración ya congelada en validación.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, two_pass = load_model(args.checkpoint, device, args.tta)
    two_pass = two_pass or args.two_pass
    cache_dir = Path(args.cache_dir)
    case_ids = [str(c).zfill(3) for c in args.cases] if args.cases else cases_of_split(cache_dir, args.split)
    print(f"{len(case_ids)} casos de {args.split} | dispositivo {device} | TTA={args.tta} | segundo pase={two_pass}"
          f" | normalizar={args.normalizar}")
    if args.normalizar:
        import normalizacion
        if not getattr(model, "canonical", False):
            print("AVISO: el modelo no es canónico; el giro se estima igual, pero el paso 1 no se aplica.")
    giros: dict[str, dict] = {}

    records: list[SliceRecord] = []
    started = time.time()
    for number, case_id in enumerate(case_ids, 1):
        case = load_case_cache(cache_dir, case_id, canonical=getattr(model, "canonical", False))
        if args.stress:
            if args.split == "test":
                raise SystemExit("La prueba de estrés es solo para validación.")
            case["images"], case["labels"] = apply_stress(case["images"], case["labels"], args.stress, mm_por_pixel(case))
        if args.normalizar:
            mm_xy = mm_por_pixel(case)
            case["images"], case["labels"], est = normalizacion.enderezar_caso(
                case["images"], case["labels"], mm_xy)
            giros[case_id] = {"angulo": round(est.angulo, 2), "corregido": est.corregir,
                              "simetria_antes": round(est.simetria_sin_girar, 3),
                              "simetria_despues": round(est.simetria, 3),
                              "mm_xy": [round(float(v), 4) for v in mm_xy]}
            print(f"    giro estimado {est.angulo:+6.1f}° "
                  f"({'corregido' if est.corregir else 'sin corregir'})", flush=True)
        case_records, predictions = evaluate_case(
            model, case_id, case["images"], case["labels"], device, args.batch_size,
            args.confidence, two_pass, keep_predictions=bool(args.save_predictions))
        records.extend(case_records)
        if predictions is not None:
            out = Path(args.save_predictions)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"{case_id}.npz", **predictions)
        print(f"  {number}/{len(case_ids)} caso {case_id}: {len(case_records)} cortes "
              f"({time.time() - started:.0f} s)", flush=True)

    seconds = time.time() - started
    result = {"checkpoint": args.checkpoint, "cache_dir": str(cache_dir), "split": args.split,
              "cases": case_ids, "stress": args.stress, "canonical": bool(getattr(model, "canonical", False)),
              "normalizar": args.normalizar, "giros": giros, "tta": args.tta, "two_pass": two_pass, "confidence": args.confidence,
              "device": str(device), "seconds": round(seconds, 1),
              "ms_per_slice": round(1000 * seconds / max(len(records), 1), 1),
              "subsets": {name: summarize(records, name) for name in SUBSETS},
              "per_case": per_case_table(records)}
    for name in ("todo", "con_hueso", "centro_15_85", "extremos"):
        print(f"\n[{name}] {result['subsets'][name]['n_slices']} cortes")
        print_table(result["subsets"][name])
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nGuardado: {args.output}")
    if args.save_records:
        Path(args.save_records).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_records, "wb") as stream:
            pickle.dump(records, stream)


if __name__ == "__main__":
    main()
