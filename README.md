# PENGWIN — Detección, segmentación y clasificación de fracturas pélvicas en CT

Proyecto Integrador Corte 2 — Analítica de Datos, Universidad Autónoma de Occidente (2026-2).
Profesor: Carlos Ferro.

Sistema propio (sin frameworks de detección/segmentación de alto nivel) para localizar
regiones óseas pélvicas en tomografía computarizada, segmentar y clasificar sus fragmentos
por hueso de pertenencia, y medir la distancia de separación de cada fragmento conminuto
respecto al fragmento principal, usando el dataset [PENGWIN](https://pengwin.grand-challenge.org/)
(MICCAI 2024–2026).

> **Uso exclusivamente académico.** Este sistema no es un dispositivo médico, no ha sido
> validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Estado actual

Semana 8 completa: dataset curado, splits fijos reproducibles, módulo de preprocesamiento
unificado, visualizador 1 (reconstrucción 3D del volumen crudo) y EDA de fragmentos.

Semana 9 completa: backbone convolucional propio con CBAM, cabeza de detección por grid,
NMS propio y prueba de overfit intencional sobre un batch pequeño. El código está en
`src/week9_detection.py` y `src/train_overfit_detection.py`; el resultado visual está en
`outputs/figures/week9_detection_overfit.png`.

Semana 10 completa: modelo propio de tres cabezas (clasificación, detección por grid con NMS
propio, segmentación semántica y de instancia) sobre un backbone compartido con CBAM, pérdida
multitarea, post-proceso 3D y distancia de separación en mm. Modelo final **H6**
(`outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth`, 2.74 M de parámetros).

| Métrica en test (15 casos) | Objetivo | Resultado |
|---|---|---|
| F1 / AUC de clasificación | 0.85 | 0.990 / 0.999 |
| IoU medio / mAP50 / mAP50:95 de detección | 0.65 / 0.65 / 0.40 | 0.943 / 0.961 / 0.902 |
| Dice / IoU de segmentación semántica | 0.85 / 0.70 | 0.971 / 0.943 |
| Dice / IoU por fragmento | 0.85 / 0.70 | 0.875 / 0.816 |

Estas cifras son de una segunda evaluación en test, hecha después de analizar los fallos de la
primera (0.845 en Dice por fragmento con la configuración congelada de antemano). Resumen en
[`docs/resumen_semana10.md`](docs/resumen_semana10.md), bitácora de decisiones en
[`docs/decisiones_semana10.md`](docs/decisiones_semana10.md) y tablas de experimentos en
[`outputs/s10/resumen_experimentos.md`](outputs/s10/resumen_experimentos.md).

Semana 11 en curso: dashboard con los tres visualizadores y demo desplegada. Ver
[`docs/semana11_dashboard.md`](docs/semana11_dashboard.md).

Detalle de la Semana 8 en [`docs/resumen_semana8.md`](docs/resumen_semana8.md).

## Estructura del repositorio

```
src/            scripts y módulos (ver comandos abajo)
tests/          pruebas automáticas (python -m unittest discover -s tests -v)
data/           manifest, splits, CSVs de EDA (data/raw/ y data/cache/ NO se versionan)
outputs/
  figures/      visualizaciones individuales (PNG, HTML)
  eda/          gráficos y tablas del análisis exploratorio
  s10/          resultados de la Semana 10 (JSON por corrida, checkpoint final, figuras)
  demo/         casos exportados para el dashboard
notebooks/      notebooks de seguimiento por semana
docs/           resúmenes, planes y bitácora de decisiones
kaggle/         lanzadores de entrenamiento en Kaggle
app/, api/      dashboard (Streamlit) y API de inferencia de la Semana 11
```

## Instalación

Requiere Python 3.11 o 3.12.

```bash
git clone https://github.com/mariavalencia30/proyecto2-analitica-datos.git
cd proyecto2-analitica-datos
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Equipos sin GPU NVIDIA (Windows):** instalar `torch` aparte antes de `requirements.txt`,
o la build CUDA por defecto de pip falla al importar por falta de drivers:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Equipos con GPU NVIDIA:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Datos

Los volúmenes CT (formato `.mha`, no NIfTI — ver `docs/resumen_semana8.md` sección 2) se
descargan de [Zenodo](https://doi.org/10.5281/zenodo.10927452) y **no se versionan**
(`data/raw/` está en `.gitignore`). Descomprimir dejando exactamente esta estructura:

```
data/raw/
├── PENGWIN_CT_train_images_part1/   001.mha …
├── PENGWIN_CT_train_images_part2/   … 100.mha
└── PENGWIN_CT_train_labels/         001.mha … 100.mha
```

## Uso

Correr **desde la raíz del repo**, en este orden (los pasos 2, 3, 6 y 7 dependen de
`data/manifest.csv` generado en el paso 1):

```bash
python src/build_manifest.py              # data/manifest.csv
python src/check_hu_outliers.py           # diagnóstico de valores HU atípicos
python src/make_splits.py                 # data/splits.json (70/15/15, seed 42)
python src/hu_windowing_check.py          # outputs/figures/hu_window_check_*.png
python src/volumen_3d_visualizer.py       # outputs/figures/volumen_3d_<id>.html
python src/eda_fragments.py               # data/eda_fragments_detalle.csv + outputs/eda/
python src/eda_fragmento_principal_y_tamano.py   # data/eda_fragmentos_individuales.csv + outputs/eda/
python src/train_overfit_detection.py --case-id 001 --epochs 150  # prueba Semana 9
python src/visualize_overfit_detection.py --case-id 001  # cajas GT vs. predicción
python src/prepare_week9_detection_data.py              # cache de cortes train/val
python src/train_week9_detection.py                      # primer entrenamiento multi-paciente
python src/visualize_week9_validation.py --case-id 014  # validación cualitativa
python -m unittest discover -s tests -v                 # pruebas Semanas 9-10
```

Semana 10 (el entrenamiento necesita GPU; la evaluación corre también en CPU):

```bash
python src/build_case_cache.py --sizes 384   # caché a 384 px por caso en data/cache/s10/cache_384
python src/train_s10.py --cache-root data/cache/s10 --size 384 --epochs 60 --base-channels 32 \
    --cbam 1 --secondary-weight 3 --augment --cosine --batch-size 16 --per-epoch 2500 --ema 0.999 \
    --seed 42 --aug-geo 1.0 --aug-rot-deg 35 --canonical \
    --output outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth          # modelo final H6
python src/evaluate_volumes.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta                   # las nueve métricas
python src/infer_volume.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta --distance-nofill --surface-metrics
                                                                             # 3D y distancia en mm
```

El modelo H6 espera los volúmenes en orientación canónica: los que la cabecera del `.mha` marca
como girados 180° se enderezan con `src/orientacion.py` antes de predecir. `evaluate_volumes.py` e
`infer_volume.py` lo hacen solos según lo que diga el checkpoint.

`src/pengwin_io.py` no se corre directo (salvo como prueba de humo, `python src/pengwin_io.py <case_id>`):
es el módulo que centraliza la carga y el preprocesamiento (clip, ventaneo HU, resize,
máscara ósea limpia) para que entrenamiento, inferencia y visualizadores usen siempre la
misma función.

La prueba de Semana 9 usa las máscaras de referencia solo para construir las cajas de
entrenamiento. Predice una caja por región visible y por corte: sacro, coxal izquierdo y
coxal derecho. La cabeza usa una celda de grid por objeto, calcula la pérdida multitarea y
aplica NMS implementado en `week9_detection.py`, sin YOLO, torchvision o Detectron2.

## Verificación de reproducibilidad

Con la semilla fija (42), `data/splits.json` debe salir idéntico al versionado en el repo,
y `data/manifest.csv` debe reportar 100/100 casos válidos.

## Restricciones del proyecto

Prohibido usar YOLO, Detectron2, Mask R-CNN preentrenado o frameworks de detección/segmentación
de alto nivel como arquitectura entregada. Transfer learning permitido únicamente en el
backbone; las cabezas se entrenan desde cero. SAM solo como baseline de comparación
zero-shot, nunca en el pipeline. Detalle completo en el enunciado oficial del curso
(`Proyecto_Corte2_PENGWIN.docx`, compartido por el profesor).

## Equipo

Ver commits y Pull Requests del repositorio.
