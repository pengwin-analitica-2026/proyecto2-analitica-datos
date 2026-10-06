"""Evalúa uno o varios checkpoints de Semana 10 sobre un split (val o test).

Usa la misma función `evaluate` del entrenamiento, así validación y test son
comparables. Con varios checkpoints promedia las probabilidades de
segmentación (ensamble); con --tta promedia también el corte espejado.
Test se toca solo con la configuración ya elegida en validación.
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
from train_week10_multitask import evaluate
from week10_multitask import PengwinMultiTaskModel

OBJETIVOS = {
    "classification_f1_macro": 0.85, "classification_auc_macro": 0.85,
    "detection_iou_mean": 0.65, "detection_map50": 0.65, "detection_map50_95": 0.40,
    "segmentation_dice_macro": 0.85, "segmentation_iou_macro": 0.70,
    "instance_dice_matched": 0.85, "instance_iou_matched": 0.70,
}


class EnsembleTTA(nn.Module):
    """Promedia las probabilidades de segmentación de varios modelos y, con
    ``tta``, también de la imagen espejada izquierda↔derecha (al volver se
    intercambian las clases de coxal). Detección y clasificación salen del
    primer modelo sin espejo. Solo inferencia; el entrenamiento no cambia."""

    SEM_SWAP = [0, 1, 3, 2]              # fondo, sacro, izq↔der
    INST_SWAP = [0, 1, 2, 5, 6, 3, 4]    # primario/secundario de izq↔der

    def __init__(self, models: list[nn.Module], tta: bool):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.tta = tta
        self.instance_mode = models[0].instance_mode
        self.in_channels = getattr(models[0], "in_channels", 1)
        self.two_pass = False

    def forward(self, x):
        out = dict(self.models[0](x))
        sem, inst, n = 0, 0, 0
        for m in self.models:
            o = m(x)
            sem = sem + o["semantic"].softmax(1)
            inst = inst + o["instance"].softmax(1)
            n += 1
            if self.tta:
                of = m(torch.flip(x, dims=(-1,)))
                s = torch.flip(of["semantic"].softmax(1), dims=(-1,))[:, self.SEM_SWAP]
                i = torch.flip(of["instance"].softmax(1), dims=(-1,))
                if i.shape[1] == 7:
                    i = i[:, self.INST_SWAP]
                sem = sem + s
                inst = inst + i
                n += 1
        out["semantic"] = (sem / n + 1e-8).log()
        out["instance"] = (inst / n + 1e-8).log()
        return out

    def train(self, mode: bool = True):
        return super().train(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="Uno o varios .pth; con varios se promedia la segmentación.")
    parser.add_argument("--tta", action="store_true", help="Promedia también el corte espejado.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--two-pass", action="store_true",
                        help="Segundo pase por región; por defecto se toma del checkpoint.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in args.checkpoint:
        ckpt = torch.load(path, map_location="cpu")
        m = PengwinMultiTaskModel(base_channels=int(ckpt["base_channels"]),
                                  instance_mode=str(ckpt.get("instance_mode", "ids")),
                                  in_channels=int(ckpt.get("in_channels", 1)))
        m.load_state_dict(ckpt["model"])
        models.append(m.to(device).eval())
    first = torch.load(args.checkpoint[0], map_location="cpu")
    if len(models) > 1 or args.tta:
        model = EnsembleTTA(models, args.tta).to(device)
        print(f"Ensamble de {len(models)} modelo(s), TTA={'sí' if args.tta else 'no'}")
        two_pass = False
    else:
        model = models[0]
        two_pass = args.two_pass or bool(first.get("two_pass", False))

    data = np.load(args.cache)
    mask = data["split_names"] == args.split
    if not mask.any():
        presentes = sorted(set(data["split_names"].tolist()))
        raise SystemExit(f"El caché no tiene split '{args.split}'. Presentes: {presentes}")
    images = torch.from_numpy(data["images"][mask].astype(np.float32))
    labels = torch.from_numpy(data["labels"][mask].astype(np.int64))
    cases = sorted(set(data["case_ids"][mask].tolist()))
    print(f"Split {args.split}: {len(images)} cortes de {len(cases)} casos {cases}")
    print(f"Segundo pase por región: {'sí' if two_pass else 'no'}")
    print()

    metrics = evaluate(model, images, labels, device, args.batch_size, args.confidence,
                       two_pass=two_pass)
    for key, value in metrics.items():
        goal = OBJETIVOS.get(key)
        if goal is None:
            flag = ""
        elif value >= goal:
            flag = "  ✅"
        else:
            flag = f"  ⚠️ objetivo {goal}"
        print(f"{key:40s} {value:.3f}{flag}")

    out = args.output or f"outputs/week10_{args.split}_metrics_{Path(args.checkpoint[0]).stem}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    resultado = {"checkpoint": args.checkpoint, "tta": args.tta, "split": args.split,
                 "cases": cases, "n_slices": int(len(images))}
    resultado.update(metrics)
    Path(out).write_text(json.dumps(resultado, indent=2))
    print()
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
