# H7 — modelo final PENGWIN (sin EMA)

Generado: 2026-10-11 20:37 · Kaggle (T4) · entrenamiento 67 min, 60 épocas.

**Receta:** igual a H6 (32 canales, CBAM 1, 60 épocas, cosine, lr 0.002, secundarios ×3, semilla 42, orientación canónica RAI,
giros ±35°) con dos cambios pedidos por el profesor: `--ema 0` (sin EMA) y `--aug-mm` (giro de entrenamiento en milímetros).

**Archivos:**
- `h7_auggeo_rot35_canon_aumm_noema_w32_e60_s42.pth`: modelo final (usar este en evaluación, demo y dashboard).
- `h7_auggeo_rot35_canon_aumm_noema_w32_e60_s42.json`: historial del entrenamiento.
- `val_H7_*.json`: validación (15 casos): sin giro, giro 60°, giro 60° + enderezado.
- `test_tercera_*.json`: **tercera evaluación en test** (una sola vez): sin enderezar y enderezado.
- `tabla_*.csv`: tablas resumen.

**No volver a entrenar ni a evaluar test**: todo queda aquí.

## Validación · sin giro
| Métrica | Valor | Objetivo | Cumple |
|---|---|---|---|
| classification_f1_macro | 0.992 | 0.85 | ✅ |
| classification_auc_macro | 0.999 | 0.85 | ✅ |
| detection_iou_mean | 0.948 | 0.65 | ✅ |
| detection_map50 | 0.974 | 0.65 | ✅ |
| detection_map50_95 | 0.903 | 0.4 | ✅ |
| segmentation_dice_macro | 0.971 | 0.85 | ✅ |
| segmentation_iou_macro | 0.944 | 0.7 | ✅ |
| instance_dice_matched | 0.873 | 0.85 | ✅ |
| instance_iou_matched | 0.812 | 0.7 | ✅ |

## Validación · giro 60° + enderezado
| Métrica | Valor | Objetivo | Cumple |
|---|---|---|---|
| classification_f1_macro | 0.988 | 0.85 | ✅ |
| classification_auc_macro | 0.999 | 0.85 | ✅ |
| detection_iou_mean | 0.924 | 0.65 | ✅ |
| detection_map50 | 0.963 | 0.65 | ✅ |
| detection_map50_95 | 0.881 | 0.4 | ✅ |
| segmentation_dice_macro | 0.956 | 0.85 | ✅ |
| segmentation_iou_macro | 0.916 | 0.7 | ✅ |
| instance_dice_matched | 0.855 | 0.85 | ✅ |
| instance_iou_matched | 0.781 | 0.7 | ✅ |

## Test (tercera evaluación) · sin enderezar
| Métrica | Valor | Objetivo | Cumple |
|---|---|---|---|
| classification_f1_macro | 0.989 | 0.85 | ✅ |
| classification_auc_macro | 0.999 | 0.85 | ✅ |
| detection_iou_mean | 0.940 | 0.65 | ✅ |
| detection_map50 | 0.962 | 0.65 | ✅ |
| detection_map50_95 | 0.899 | 0.4 | ✅ |
| segmentation_dice_macro | 0.971 | 0.85 | ✅ |
| segmentation_iou_macro | 0.944 | 0.7 | ✅ |
| instance_dice_matched | 0.879 | 0.85 | ✅ |
| instance_iou_matched | 0.819 | 0.7 | ✅ |

## Test (tercera evaluación) · enderezado (pipeline final)
| Métrica | Valor | Objetivo | Cumple |
|---|---|---|---|
| classification_f1_macro | 0.989 | 0.85 | ✅ |
| classification_auc_macro | 0.999 | 0.85 | ✅ |
| detection_iou_mean | 0.938 | 0.65 | ✅ |
| detection_map50 | 0.958 | 0.65 | ✅ |
| detection_map50_95 | 0.893 | 0.4 | ✅ |
| segmentation_dice_macro | 0.968 | 0.85 | ✅ |
| segmentation_iou_macro | 0.938 | 0.7 | ✅ |
| instance_dice_matched | 0.874 | 0.85 | ✅ |
| instance_iou_matched | 0.812 | 0.7 | ✅ |
