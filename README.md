# PENGWIN —- Detección, segmentación y clasificación de fracturas pélvicas en CT

Proyecto Integrador Corte 2 — Analítica de Datos, Universidad Autónoma de Occidente (2026-2).
Profesor: Carlos Ferro.

Sistema propio, sin frameworks de detección ni segmentación de alto nivel, que en tomografías de
pelvis localiza las regiones óseas (sacro, coxal izquierdo, coxal derecho), segmenta y clasifica sus
fragmentos de fractura, y mide en milímetros la distancia de cada fragmento al fragmento principal
de su hueso. Usa el dataset [PENGWIN 2024](https://pengwin.grand-challenge.org/).

> **Uso exclusivamente académico.** Este sistema no es un dispositivo médico, no ha sido validado
> clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Resultado final

El modelo entregado es **H8** (`outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth`): backbone propio de 32 canales
con CBAM, tres cabezas (clasificación, detección con grid y NMS propios, segmentación semántica e
instancia) y orientación canónica de los volúmenes según su cabecera. Frente a H6, el modelo
anterior, cambia el entrenamiento en tres puntos: **sin EMA**, **giro del aumento en milímetros
reales** (±35°) y **aumento de ruido gaussiano**.

| Métrica (15 casos, H8 semilla 42) | Objetivo | Validación | Test |
|---|---|---|---|
| F1 / AUC de clasificación | 0.85 / 0.85 | 0.991 / 0.999 | 0.989 / 0.999 |
| IoU medio / mAP50 / mAP50:95 de detección | 0.65 / 0.65 / 0.40 | 0.946 / 0.969 / 0.900 | 0.938 / 0.953 / 0.898 |
| Dice / IoU de segmentación semántica | 0.85 / 0.70 | 0.970 / 0.943 | 0.968 / 0.939 |
| Dice / IoU por fragmento | 0.85 / 0.70 | 0.865 / 0.800 | 0.874 / 0.812 |

Con tres semillas (42, 43 y 44) el Dice por fragmento en test es 0.874, 0.875 y 0.877
(0.875 ± 0.001), y las tres cumplen los nueve objetivos.

H8 se eligió entre cinco variantes por ser la más robusta al ruido gaussiano: con σ = 10 y 25
niveles de gris mantiene 0.860 y 0.835 de Dice por fragmento en validación, donde las demás caen a
0.72 y 0.23 como mucho. No es robusto a todo: con ruido fuerte y con un giro de 60° queda por debajo
del objetivo (model card, sección 4.3). La cifra de test es la **sexta consulta a test** del proyecto; la
primera evaluación limpia fue D5 (0.845). Cómo se eligió H8, las ablaciones, la robustez y las
limitaciones están en el [model card](MODEL_CARD.md) y en [`docs/modelo_h8.md`](docs/modelo_h8.md).

## Documentación

| Documento | Contenido |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Modelo, datos, métricas, ablaciones, protocolo de test y limitaciones |
| [`docs/modelo_h8.md`](docs/modelo_h8.md) | Modelo entregado H8: qué cambia frente a H6, ficha técnica y resultados en validación y test |
| [`docs/experimentos_ema_rotacion_ruido.md`](docs/experimentos_ema_rotacion_ruido.md) | Comparación de cinco modelos tras la asesoría (EMA, giro en milímetros y ruido), ablación de CBAM por cabeza y HD95/ASSD por fragmento |
| [`docs/mini_paper/Mini_paper_PENGWIN.pdf`](docs/mini_paper/Mini_paper_PENGWIN.pdf) | **Informe técnico corto con estructura de mini-paper** (entregable, 8 páginas): decisiones de arquitectura, ablación, matriz de confusión, cinco peores casos y distancia predicha contra la real. Fuente en `.docx` y `build.js` |
| [`docs/informe/Informe_tecnico_PENGWIN.docx`](docs/informe/Informe_tecnico_PENGWIN.docx) | Versión extendida del informe (21 páginas), con el preprocesamiento, la investigación del Dice por fragmento y el trabajo por semana |
| [`IA_USAGE.md`](IA_USAGE.md) | Registro y análisis crítico del uso de IA generativa |
| [`docs/resumen_semana8.md`](docs/resumen_semana8.md) | Dataset, preprocesamiento, EDA y Visualizador 1 |
| [`docs/resumen_semana9.md`](docs/resumen_semana9.md) | Backbone, CBAM, detección y overfit intencional |
| [`docs/decisiones_semana10.md`](docs/decisiones_semana10.md) | Semana 10: bitácora de experimentos, de la línea base a H6, y las dos evaluaciones en test |
| [`outputs/s10/resumen_experimentos.md`](outputs/s10/resumen_experimentos.md) | Semana 10: tabla de todas las corridas y dónde está cada archivo de resultados |
| [`docs/semana11_dashboard.md`](docs/semana11_dashboard.md) | Semana 11: dashboard con los tres visualizadores, orientación y distancia |
| [`docs/despliegue_aws_streamlit.md`](docs/despliegue_aws_streamlit.md) | Semana 11: despliegue en Streamlit Cloud y AWS Lambda |
| [`docs/cicd_github_actions.md`](docs/cicd_github_actions.md) | Semana 11: pruebas y despliegue automáticos |
| [`docs/cuantizacion.md`](docs/cuantizacion.md) | Cuantización de H8: qué es, int8 para CPU y fp16 para GPU, resultados en validación y test, y cómo se elige la precisión en la API y el dashboard |
| [`docs/video/demo_computador.mp4`](docs/video/demo_computador.mp4) | Semana 11: recorrido completo del dashboard publicado, grabado en el computador (110 s, con audio): inferencia corte a corte, tres visualizadores, métricas, latencia y tabla de distancias |
| [`docs/video/demo_celular.mp4`](docs/video/demo_celular.mp4) | Semana 11: video del dashboard publicado, grabado desde un celular |
| `notebooks/01` a `05` | Seguimiento por semana: EDA, intensidades, Semana 9, Semana 10 y casos difíciles (el 05 se hizo con D5) |
| `notebooks/11_modelo_h8.ipynb` | Modelo entregado H8: entrenamiento, métricas en validación y test, comparación de modelos, CBAM, HD95/ASSD y distancia |

Cada tema está en un solo documento. Los resultados del modelo están en el model card y no se
repiten en los demás.

## Estructura del repositorio

```
src/        scripts y módulos (ver los comandos abajo)
tests/      pruebas automáticas
notebooks/  notebooks de seguimiento por semana
kaggle/     kernels de entrenamiento en GPU (uno por receta; los de H8 en kaggle/h8/)
colab/      genera el cuaderno de Colab que entrena H6 (notebooks/colab/colab_s10.ipynb)
app/        dashboard Streamlit
api/        API de inferencia
deploy/     scripts de despliegue
data/       manifest, splits y CSVs del EDA (data/raw/ y data/cache/ NO se versionan)
outputs/    figuras (figures/), EDA (eda/), resultados y checkpoints de la Semana 10 (s10/),
            resultados de H8 (h8/) y casos exportados para el dashboard (demo/)
docs/       un documento por semana o por tema, informe técnico y mini-paper
```

## Instalación

Requiere Python 3.11 o 3.12.

```bash
git clone https://github.com/pengwin-analitica-2026/proyecto2-analitica-datos.git
cd proyecto2-analitica-datos
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows sin GPU NVIDIA:** instalar `torch` aparte antes de `requirements.txt`; la build CUDA
por defecto falla al importar si no hay drivers.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Con GPU NVIDIA:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Datos

Los volúmenes están en formato `.mha` (no NIfTI; el profesor aprobó este formato). Se descargan de
[Zenodo](https://doi.org/10.5281/zenodo.10927452) (licencia CC BY 4.0) y **no se versionan**. Descomprimir con esta
estructura exacta:

```
data/raw/
├── PENGWIN_CT_train_images_part1/   001.mha …
├── PENGWIN_CT_train_images_part2/   … 100.mha
└── PENGWIN_CT_train_labels/         001.mha … 100.mha
```

Todos los comandos se corren **desde la raíz del repo**, con el entorno activo.

## 1. Datos y EDA (Semana 8)

El paso 1 va primero: los pasos 2, 3, 6 y 7 leen `data/manifest.csv`.

```bash
python src/build_manifest.py                      # 1. data/manifest.csv
python src/check_hu_outliers.py                   # 2. diagnóstico de HU atípicos
python src/make_splits.py                         # 3. data/splits.json (70/15/15, seed 42)
python src/hu_windowing_check.py                  # 4. outputs/figures/hu_window_check_*.png
python src/volumen_3d_visualizer.py               # 5. Visualizador 1 (3D): outputs/figures/volumen_3d_<id>.html
python src/eda_fragments.py                       # 6. data/eda_fragments_detalle.csv + outputs/eda/
python src/eda_fragmento_principal_y_tamano.py    # 7. data/eda_fragmentos_individuales.csv + outputs/eda/
```

El MIP del Visualizador 1 (`src/raw_view.py`) se ve en la sección 6.1 del notebook 01 y en el
dashboard.

`src/pengwin_io.py` es el módulo que centraliza la carga y el preprocesamiento (clip, ventana ósea,
gamma, resize y máscara ósea limpia). No se corre directo, salvo como prueba de humo:
`python src/pengwin_io.py 001`.

## 2. Detección (Semana 9)

```bash
python src/train_overfit_detection.py --case-id 001 --epochs 150   # overfit intencional
python src/visualize_overfit_detection.py --case-id 001            # cajas GT contra predicción
python src/prepare_week9_detection_data.py                          # caché de cortes train/val
python src/train_week9_detection.py                                 # primer entrenamiento multi-paciente
python src/visualize_week9_validation.py --case-id 014              # validación cualitativa
```

## 3. Modelo final H8

H8 es la receta de H6 (Semana 10) con tres cambios en el entrenamiento. Detalle en
[`docs/modelo_h8.md`](docs/modelo_h8.md).

### 3.1 Caché por caso

Guarda cada corte de cada caso a 384 px (ventana ósea + gamma) en `data/cache/s10/`:

```bash
python src/build_case_cache.py --sizes 384 --modes stretch --workers 4
```

Deja `data/cache/s10/cache_384/<caso>.npz`, `labels_native/` e `index.csv`.

### 3.2 Entrenamiento

Requiere GPU. H8 se entrenó en Kaggle con una T4 (67 minutos); los kernels están en `kaggle/h8/`
(`python kaggle/h8/armar.py --salida <carpeta> --usuario <usuario>` los genera):

```bash
python src/train_s10.py --cache-root data/cache/s10 --size 384 --mode stretch --epochs 60 \
    --batch-size 16 --lr 0.002 --base-channels 32 --cbam 1 --instance-mode primary --augment --cosine \
    --margin 10 --neg-fraction 0.15 --per-epoch 2500 --val-stride 6 --eval-every 10 --max-minutes 170 \
    --secondary-weight 3.0 --canonical --aug-geo 1.0 --aug-rot-deg 35 --aug-mm --ema 0 \
    --aug-noise 0.1 --seed 42 --output outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth
```

Frente a H6 cambian tres opciones: `--ema 0` (sin promedio móvil de los pesos), `--aug-mm` (el giro
del aumento se define en milímetros, `src/aumento_fisico.py`) y `--aug-noise 0.1` (ruido gaussiano en
el 30 % de las muestras). `--canonical` endereza los 34 volúmenes guardados en orientación LPI y
queda registrado en el checkpoint, así que la evaluación lo aplica sola.

El checkpoint entregado ya está en el repo, así que este paso se puede saltar.

### 3.3 Evaluación sobre el volumen completo

```bash
python src/evaluate_volumes.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta \
    --output outputs/h8/validacion/val_h8_base.json
```

Resultado esperado en validación: Dice por fragmento 0.865 (puede variar en la tercera cifra según
el hardware). Reporta la detección cruda (cabeza grid) y la refinada, y separa el Dice en fragmento
primario y secundario.

Robustez (solo validación): `--stress noise=10`, `--stress noise=25`, `--stress rotmm=60` y
`--stress rotmm=60 --normalizar` (giro con enderezado previo).

HD95 y ASSD por fragmento, con emparejamiento por IoU, no detectados, medias, medianas y cobertura:

```bash
python src/metricas_superficie.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta \
    --output outputs/h8/validacion/superficie_val_h8.json
```

Distancia de separación en mm por fragmento, sobre la predicción y sobre el ground truth:

```bash
python src/infer_volume.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta --distance-nofill --surface-metrics \
    --output outputs/h8/validacion/distancias_val_h8.json
```

Grosor de corte (solo validación): `--zstep 2` o `--zstep 3` conserva 1 de cada 2 o 3 cortes, y
`--umbral-mm` fija los tamaños mínimos del post-proceso 3D en mm³ en vez de vóxeles; valen para
`infer_volume.py` y `metricas_superficie.py`. Resultados en `outputs/h8/validacion/grosor_corte/`.

**Test:** el protocolo es evaluar una sola vez con la configuración congelada. La configuración de H8
está en `outputs/h8/test/CONFIG_CONGELADA_TEST_H8.md` y sus resultados en `outputs/h8/test/`. No se
recomienda volver a correr test para ajustar nada.

### 3.4 Línea base SAM (solo comparación)

SAM ViT-B no forma parte del sistema: recibe como prompt la caja que predice nuestro detector y se
compara con nuestra máscara semántica. Con H8, en test, SAM da 0.889 de Dice por región y H8 0.931
(`outputs/h8/test/sam_baseline_test_h8.json`; la medición anterior con H6 está en
`outputs/s10/sam_baseline_test_h6.json`). Pesos: `sam_vit_b_01ec64.pth`, del repositorio oficial de
Segment Anything.

```bash
python src/sam_baseline.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --sam-checkpoint <ruta>/sam_vit_b_01ec64.pth --cache-dir data/cache/s10/cache_384 \
    --split test --stride 3 --output outputs/h8/test/sam_baseline_test_h8.json
```

El enunciado pide la comparación sobre test, por eso `--split test` (sin esa opción usa validación).

### 3.5 Pruebas automáticas

```bash
python -m unittest discover -s tests -v
```

## 4. Demo y visualizadores (Semana 11)

`demo_export.py` corre el pipeline completo una vez por caso (de 4 a 25 minutos en CPU con TTA,
según la máquina y el número de cortes) y deja todo en `outputs/demo/<modelo>/<caso>/`. El
dashboard solo lee esos archivos. Casos versionados, todos con H8 y TTA: 024 (validación), 044 y 099
(validación, guardados girados 180°) y 039 (test).

```bash
python src/demo_export.py --case 024 044 --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth --tta
streamlit run app/dashboard.py
```

| Visualizador | Qué muestra |
|---|---|
| 1. Volumen crudo | MIP axial, coronal y sagital del hueso (umbral de 250 HU, sin camilla ni cables) e isosuperficie 3D. Sin modelo |
| 2. Corte por corte | Cajas, máscaras por hueso, ground truth opcional e ID y distancia en mm de cada fragmento |
| 3. Reconstrucción 3D | Una malla por fragmento, con su ID y su distancia en mm al principal |

La demo endereza además el giro axial del paciente si es de 10° o más (`--sin-enderezar` lo
omite) y fija los tamaños mínimos del post-proceso 3D en mm³ (`--umbral-voxeles` vuelve a los 30 y
200 vóxeles de la evaluación oficial). Con H8 la demo endereza los volúmenes guardados girados 180° (cabecera `TransformMatrix = -1 0 0 0
-1 0 0 0 1`) antes de predecir, una sola vez y para las tres vistas y el ground truth; el dashboard
indica en la barra lateral si el caso se enderezó. Detalle en
[`docs/semana11_dashboard.md`](docs/semana11_dashboard.md).

### Latencia por imagen

```bash
python src/measure_latency.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --volume outputs/demo/h8_aumm_noise01_noema_w32_e60_s42/024/volume.npz \
    --device cpu --device cuda
```

Mide un corte a la vez y reporta la mediana y el percentil 90. Escribe en
`outputs/demo/latency.json`, que el dashboard muestra. Si no hay GPU, omite `cuda` y lo avisa. Los
resultados están en el [model card](MODEL_CARD.md), sección 4.8.

**Cuantización.** El mismo H8 corre en int8 en CPU (2.3 veces más rápido) y en fp16 en GPU (1.8 veces),
sin perder Dice ([`docs/cuantizacion.md`](docs/cuantizacion.md)). La API elige sola con
`PRECISION=auto` y el dashboard deja escoger dónde corre el modelo y con qué precisión.

### Despliegue

El dashboard está publicado en **Streamlit Community Cloud** (app `app/dashboard.py`, rama `main`) y
la inferencia en vivo corre en una **API en AWS Lambda**, cuya imagen copia el checkpoint de H8 y lo sirve en int8. Cada push a `main` corre las
pruebas y vuelve a desplegar los dos (GitHub Actions para la API, con OIDC y sin claves guardadas;
Streamlit Cloud para el dashboard). Detalle en
[`docs/despliegue_aws_streamlit.md`](docs/despliegue_aws_streamlit.md) y
[`docs/cicd_github_actions.md`](docs/cicd_github_actions.md).

## Verificación de reproducibilidad

- Con la semilla 42, `data/splits.json` sale idéntico al versionado, y `data/manifest.csv` reporta
  100 de 100 casos válidos.
- La evaluación de la sección 3.3 sobre el checkpoint entregado debe reproducir el JSON de
  validación versionado en `outputs/h8/validacion/`.

## Restricciones del proyecto

Prohibido usar YOLO, Detectron2, Mask R-CNN preentrenado o frameworks de detección o segmentación de
alto nivel. Transfer learning solo en el backbone; las cabezas se entrenan desde cero. SAM solo
como línea base zero-shot, nunca en el pipeline. Detalle en el enunciado oficial del curso.

## Equipo

| Integrante | Código |
|---|---|
| Gustavo Adolfo Chipantiza | 2222022 |
| Federico Gonzales Cárdenas | 2225657 |
| Juan Jacobo Delgado Melo | 2230110 |
| María de los Ángeles Valencia Hurtado | 2235112 |
