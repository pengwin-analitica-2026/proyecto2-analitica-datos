"""Tabla comparativa de corridas evaluadas con evaluate_volumes.py.

    python src/comparar_corridas.py outputs/s10/linea_base_val.json outputs/s10/b0_cbam1_s42_val.json ... [--subset todo]
"""
import argparse, json
from pathlib import Path

FILAS = [("classification_f1_macro", "Clasif. F1"), ("classification_auc_macro", "Clasif. AUC"),
         ("detection_iou_mean", "Det. IoU (refinada)"), ("detection_map50", "Det. mAP50 (refinada)"),
         ("detection_map50_95", "Det. mAP50:95 (refinada)"), ("detection_raw_map50", "Det. mAP50 (cruda)"),
         ("detection_raw_map50_95", "Det. mAP50:95 (cruda)"), ("segmentation_dice_macro", "Seg. Dice"),
         ("segmentation_iou_macro", "Seg. IoU"), ("instance_dice_matched", "Inst. Dice"),
         ("instance_iou_matched", "Inst. IoU"), ("instance_dice_primary", "Inst. Dice principal"),
         ("instance_dice_secondary", "Inst. Dice secundario"), ("instance_missed_secondary", "Secundarios perdidos")]

def main():
    p = argparse.ArgumentParser(); p.add_argument("files", nargs="+"); p.add_argument("--subset", default="todo")
    a = p.parse_args()
    runs = [(Path(f).stem.replace("_val", ""), json.load(open(f))["subsets"][a.subset]) for f in a.files]
    print("| Métrica | " + " | ".join(n for n, _ in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for key, label in FILAS:
        vals = [r.get(key) for _, r in runs]
        print(f"| {label} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")

if __name__ == "__main__":
    main()
