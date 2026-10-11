"""Análisis de fallos por caso: por qué un volumen sale mal (solo diagnóstico, no ajusta nada).

Para cada caso pedido calcula el Dice semántico por región, la matriz de confusión entre
regiones (para ver intercambios izquierda/derecha u hueso tomado por otra región), cuánta
predicción cae fuera del hueso etiquetado, y guarda paneles CT / real / predicción en cortes
repartidos a lo largo del hueso.

Uso:
    python src/diagnostico_casos.py --checkpoint outputs/s10/d5_ema_w32_e60_s42.pth \
        --cache-dir data/cache/s10/cache_384 --cases 087 078 --tta --output-dir outputs/s10/diag_casos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisis_instancias import as3d, match_slice, predict_case, save_example  # noqa: E402
from build_case_cache import load_case_cache  # noqa: E402
from evaluate_volumes import PRIMARY_IDS, load_model, semantic_from_labels  # noqa: E402
from instance_postprocess import primary_secondary_to_instances  # noqa: E402

NAMES = ("fondo", "sacro", "coxal_izq", "coxal_der")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--panels", type=int, default=6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device, args.tta)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for case_id in args.cases:
        case = load_case_cache(args.cache_dir, case_id, canonical=getattr(model, "canonical", False))
        images, labels = case["images"], case["labels"]
        sem, ps, _ = predict_case(model, images, device)
        inst = as3d(primary_secondary_to_instances(ps, sem))
        gt = semantic_from_labels(labels)
        confusion = np.bincount(gt.ravel().astype(np.int64) * 4 + sem.ravel().astype(np.int64),
                                minlength=16).reshape(4, 4)
        dice = {NAMES[c]: round(float(2 * confusion[c, c] / max(confusion[c].sum() + confusion[:, c].sum(), 1)), 3)
                for c in (1, 2, 3)}
        bone_z = np.flatnonzero((labels != 0).any(axis=(1, 2)))
        per_slice = []
        for z in bone_z:
            rows, extra = match_slice(inst[z], labels[z])
            per_slice.append((int(z), float(np.mean([r[1] for r in rows])), len(rows), extra))
        frag = [r for z in bone_z for r in match_slice(inst[z], labels[z])[0]]
        report[case_id] = {
            "cortes": int(len(images)), "cortes_con_hueso": int(len(bone_z)),
            "intensidad_media_u8": round(float(images.mean()), 1),
            "intensidad_p99_u8": float(np.percentile(images, 99)),
            "fraccion_pixeles_hueso_real": round(float((gt > 0).mean()), 4),
            "dice_semantico": dice,
            "confusion_real_x_pred (filas real, columnas pred)": {NAMES[i]: {NAMES[j]: int(confusion[i, j]) for j in range(4)} for i in range(4)},
            "hueso_real_predicho_fondo": round(float(confusion[1:, 0].sum() / max(confusion[1:].sum(), 1)), 3),
            "prediccion_sobre_fondo_real": round(float(confusion[0, 1:].sum() / max(confusion[:, 1:].sum(), 1)), 3),
            "hueso_asignado_a_otra_region": round(float((confusion[1:, 1:].sum() - np.trace(confusion[1:, 1:])) / max(confusion[1:].sum(), 1)), 3),
            "dice_instancias": round(float(np.mean([r[1] for r in frag])), 3),
            "dice_primario": round(float(np.mean([r[1] for r in frag if r[0] in PRIMARY_IDS])), 3),
            "fragmentos_predichos_sobrantes": int(sum(p[3] for p in per_slice)),
            "peores_cortes": sorted(per_slice, key=lambda p: p[1])[:8],
        }
        picks = bone_z[np.linspace(0, len(bone_z) - 1, args.panels + 2).astype(int)[1:-1]]
        for z in picks:
            score = next(p[1] for p in per_slice if p[0] == z)
            save_example(out_dir / f"{case_id}_z{int(z):03d}.png", images[z], labels[z], inst[z],
                         f"caso {case_id} corte {int(z)} ({np.searchsorted(bone_z, z) / max(len(bone_z) - 1, 1):.0%} del hueso) | Dice instancias del corte {score:.2f}")
        print(case_id, json.dumps({k: report[case_id][k] for k in (
            "dice_semantico", "hueso_real_predicho_fondo", "prediccion_sobre_fondo_real",
            "hueso_asignado_a_otra_region", "dice_instancias", "fragmentos_predichos_sobrantes")}, ensure_ascii=False), flush=True)
    (out_dir / "diagnostico.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
