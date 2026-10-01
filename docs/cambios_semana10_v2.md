# Semana 10 v2 — cambios de código (fusionados sobre `continuacion_semana10_fixed`)

Todo controlado por flags; sin flags, el comportamiento es el original (31 IDs).

| Archivo | Cambio |
|---|---|
| `src/week10_multitask.py` | `instance_mode` `ids` (31 canales) / `primary` (7: fondo + primario/secundario por región, formulación del challenge PENGWIN 2024). `labels_to_primary_secondary`, `flip_sagittal`, `fracture_distance_weight` (FracSegNet), `region_crops` / `two_pass_instances`. `multitask_loss` acepta `fracture_weight`, `secondary_weight`, mapa precomputado. `in_channels` para entrada 2.5D. |
| `src/train_week10_multitask.py` | Flags `--instance-mode`, `--augment`, `--fracture-weight`, `--secondary-weight`, `--cosine`, `--crop-prob`, `--two-pass`. `--resume` reinicializa solo capas con forma distinta. Checkpoint parcial `*_last.pth` en cada evaluación. Mapa de fractura precomputado por corte. |
| `src/instance_postprocess.py` | `primary_secondary_to_instances` (primario = componente mayor, secundarios por conectividad; opciones `fill_holes`, `absorb_small`). |
| `src/evaluate_week10.py` (nuevo) | Evalúa uno o varios checkpoints en val/test con la misma función del entrenamiento; `--tta` (espejo con intercambio de coxales). |
| `src/inference_week10.py` | Lee `instance_mode`, `two_pass`, `in_channels` del checkpoint. |
| `src/calibrate_primary_postprocess.py` (nuevo) | Barrido del postproceso sobre validación. |
| `src/error_analysis_instances.py` (nuevo) | Dice por fragmento según tamaño, tipo de fallo y región; figura de peores cortes. |
| `src/prepare_week9_detection_data.py` | `--extent`, `--negatives-per-case` (cortes sin hueso), `--context K` (2.5D). |
| `src/make_test_fullrange.py` (nuevo) | Caché de test con cortes de todo el volumen (para AUC de clasificación). |
| `src/pengwin_metrics.py` | Compatible con NumPy 1 y 2. |
| `tests/test_primary_secondary.py` (nuevo), `tests/test_week10_multitask.py` | 9 pruebas nuevas; prueba vieja actualizada. 37 pruebas en total en verde. |

## Receta ganadora (con CLAHE; a repetir con gamma)

```bash
# cachés (Mac)
python src/prepare_week9_detection_data.py --max-train-cases 70 --max-val-cases 15 --slices-per-case 12 --output data/cache/week10_slices_g.npz
python src/prepare_week9_detection_data.py --max-train-cases 0 --max-val-cases 0 --max-test-cases 15 --slices-per-case 12 --output data/cache/week10_test_g.npz
python src/prepare_week9_detection_data.py --max-train-cases 70 --max-val-cases 15 --slices-per-case 12 --size 384 --output data/cache/week10_slices_384_g.npz
python src/prepare_week9_detection_data.py --max-train-cases 0 --max-val-cases 0 --max-test-cases 15 --slices-per-case 12 --size 384 --output data/cache/week10_test_384_g.npz

# GPU: base (80 ép.) → ajuste fino con pesos (30 ép.) → ajuste fino a 384 (20 ép.) → TTA
python src/train_week10_multitask.py --cache data/cache/week10_slices_g.npz --instance-mode primary --augment --cosine --base-channels 16 --fracture-weight 4 --epochs 80 --eval-every 10 --output outputs/w10g_base.pth --metrics-output outputs/w10g_base_val.json
python src/train_week10_multitask.py --cache data/cache/week10_slices_g.npz --instance-mode primary --augment --cosine --base-channels 16 --fracture-weight 8 --secondary-weight 3 --resume outputs/w10g_base.pth --lr 5e-4 --epochs 30 --eval-every 10 --output outputs/w10g_sec3.pth --metrics-output outputs/w10g_sec3_val.json
python src/train_week10_multitask.py --cache data/cache/week10_slices_384_g.npz --instance-mode primary --augment --cosine --base-channels 16 --fracture-weight 8 --secondary-weight 3 --resume outputs/w10g_sec3.pth --lr 5e-4 --epochs 20 --eval-every 5 --output outputs/w10g_384.pth --metrics-output outputs/w10g_384_val.json
python src/evaluate_week10.py --cache data/cache/week10_slices_384_g.npz --split val  --tta --checkpoint outputs/w10g_384.pth --output outputs/w10g_384_val_tta.json
python src/evaluate_week10.py --cache data/cache/week10_test_384_g.npz  --split test --tta --checkpoint outputs/w10g_384.pth --output outputs/w10g_384_test_tta.json
python src/error_analysis_instances.py --cache data/cache/week10_slices_384_g.npz --checkpoint outputs/w10g_384.pth --out-dir outputs/error_analysis_g
```

Referencia con CLAHE (test, 180 cortes): Dice instancia 0,814 / IoU 0,747 (384 + TTA); sec3 a 256: 0,807 / 0,733.
