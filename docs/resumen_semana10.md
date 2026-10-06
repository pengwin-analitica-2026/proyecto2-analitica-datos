# Semana 10 — pipeline multitarea (cierre, 6 de octubre de 2026)

Resumen de lo que quedó. El detalle de cada experimento y de cada decisión está en
[`decisiones_semana10.md`](decisiones_semana10.md) (bitácora), las tablas completas en
[`../outputs/s10/resumen_experimentos.md`](../outputs/s10/resumen_experimentos.md) y el recorrido
ejecutable en `notebooks/04_semana10_pipeline_multitarea.ipynb`.

Este documento reemplaza la versión del 21 de septiembre, que describía el primer intento de la
Semana 10 (modelos `w10g_*`). Ese intento quedó como línea base.

## Qué se construyó

Un solo modelo propio con backbone compartido y tres cabezas, entrenado desde cero sobre los 70
casos de entrenamiento y evaluado sobre el volumen completo de cada caso.

| Pieza | Cómo quedó | Archivo |
|---|---|---|
| Backbone | Convolucional propio, 32 canales base, CBAM en una etapa antes de bifurcar. 2.74 M de parámetros en total | `src/week9_detection.py`, `src/week10_multitask.py` |
| Clasificación | Qué regiones hay en el corte (sacro, coxal izquierdo, coxal derecho) | `ClassificationHead` |
| Detección | Grid propio con una celda por región y NMS propio. Se reporta la caja cruda y la refinada con la semántica | `GridDetectionHead`, `pengwin_metrics.py` |
| Segmentación semántica | Fondo y las tres regiones | `SegmentationHead` |
| Segmentación de instancia | Por región, fragmento principal o secundario; los fragmentos salen por componentes conexas (`min_size` 12) | `instance_postprocess.py` |
| Pérdida | Compuesta, con λ por término (detección 1, clasificación 0.5, semántica 2, instancia 1, borde 1) y peso ×3 a los secundarios | `train_s10.py` |
| Volumen | Post-proceso 3D por región y distancia mínima borde a borde en mm con el spacing de la cabecera | `volume_postprocess.py`, `fragment_distances.py`, `infer_volume.py` |
| Evaluación | Las nueve métricas sobre todos los cortes del caso, con subconjuntos (centro, extremos) y tabla por caso | `evaluate_volumes.py` |

No se usa YOLO, Detectron2, Mask R-CNN ni ningún framework de detección o segmentación. SAM solo
aparece como línea base de comparación.

## Modelo final: H6

Receta: 384 px, 60 épocas de 2 500 cortes, lote 16, coseno, EMA 0.999, más dos cambios que salieron
del análisis de fallos de test:

1. **Orientación canónica.** 34 de los 100 volúmenes están guardados girados 180° en el plano; lo
   dice la cabecera del `.mha`. Se enderezan antes de entrar a la red, en entrenamiento y en
   inferencia (`src/orientacion.py`, bandera `--canonical`). Es una regla fija, no una cabeza nueva.
2. **Aumento geométrico.** Rotación de ±35°, escala, proporción y desplazamiento al entrenar
   (`--aug-geo 1.0 --aug-rot-deg 35`).

Checkpoint: `outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth`.

## Resultados

Test, 15 casos, 4 664 cortes, un modelo con TTA espejo.

| Métrica | Objetivo | Test, primera evaluación (D5) | Test, segunda evaluación (H6) |
|---|---|---|---|
| F1 de clasificación | 0.85 | 0.984 | 0.990 |
| AUC de clasificación | 0.85 | 0.998 | 0.999 |
| IoU medio de detección | 0.65 | 0.898 | 0.943 |
| mAP50 de detección | 0.65 | 0.920 | 0.961 |
| mAP50:95 de detección | 0.40 | 0.853 | 0.902 |
| Dice de segmentación semántica | 0.85 | 0.949 | 0.971 |
| IoU de segmentación semántica | 0.70 | 0.904 | 0.943 |
| **Dice por fragmento** | **0.85** | 0.845 | **0.875** |
| IoU por fragmento | 0.70 | 0.781 | 0.816 |

**Cómo leer las dos columnas.** La primera evaluación se hizo con la configuración congelada antes
de ver test y es la cifra limpia. La segunda se hizo después de analizar los fallos de la primera,
y ese análisis miró dos casos de test (012 y 039); el modelo, eso sí, se eligió solo con validación
y con pruebas de estrés. Las tres semillas de H6 dan 0.875, 0.875 y 0.876 en test; el ensamble,
0.879.

Validación (15 casos): H6 da 0.869 / 0.868 / 0.872 con tres semillas; D5 daba 0.851 / 0.855 / 0.854.
Con la validación girada 30°, H6 da 0.854 y D5 0.559.

## Ablaciones, SAM, distancia y latencia

- **CBAM (con y sin):** aporta en clasificación (F1 +0.013) y en detección refinada (mAP50:95
  +0.011); la segmentación queda igual.
- **Transfer learning (backbone preentrenado en Intel Image, con y sin):** sin ganancia consistente.
- **λ:** duplicar instancia y borde no mejora y empeora la detección cruda; se dejan los valores por
  defecto. Las tres ablaciones se midieron sobre la receta base (c2 y D5), no sobre H6.
- **SAM ViT-B zero-shot con la caja predicha como prompt (test, con H6):** Dice 0.889 por región,
  frente a 0.934 de nuestra máscara. SAM no separa ni clasifica fragmentos.
- **Distancia en mm (test, H6):** error medio 1.28 mm, correlación 0.82. Un solo fragmento (caso
  007) concentra el 79 % del error.
- **Latencia por corte a 384 px:** 64 ms en GPU L4 y 162 ms en CPU de 12 hilos con un modelo; 72 y
  250 ms con TTA.

## Limitaciones que hay que declarar

- El Dice de los fragmentos secundarios es 0.750 en test: es la parte más débil.
- Fragmentos en contacto miden 0 mm y pueden fusionarse. Solo 5 de los 41 secundarios de validación
  están separados en el ground truth.
- El caso 039 (0.784) sigue siendo el peor de test y el 087 (0.790) el peor de validación.
- El modelo necesita la orientación de la cabecera. Un volumen sin esa información, o con la
  cabecera mal, entraría en la orientación equivocada.
- Uso exclusivamente académico; no es un dispositivo médico.

## Cómo reproducir

```bash
python -m unittest discover -s tests -v
python src/build_case_cache.py --sizes 384        # caché a 384 px por caso en data/cache/s10/cache_384
python src/train_s10.py --cache-root data/cache/s10 --size 384 --epochs 60 --base-channels 32 \
    --cbam 1 --secondary-weight 3 --augment --cosine --batch-size 16 --per-epoch 2500 --ema 0.999 \
    --seed 42 --aug-geo 1.0 --aug-rot-deg 35 --canonical \
    --output outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth
python src/evaluate_volumes.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta
python src/infer_volume.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta --distance-nofill --surface-metrics
```

El entrenamiento necesita GPU: D5 tardó unos 65 minutos en una T4 de Kaggle, y cada semilla de H6
entre 97 y 124 minutos en una L4 de Colab con dos entrenamientos a la vez. La evaluación corre
también en CPU.
