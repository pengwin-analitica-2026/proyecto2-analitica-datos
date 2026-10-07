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

El modelo entregado es **H6** (`outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth`): backbone
propio de 32 canales con CBAM, tres cabezas (clasificación, detección con grid y NMS propios,
segmentación semántica e instancia), orientación canónica de los volúmenes según su cabecera,
aumento geométrico con rotación de ±35° y pesos EMA.

| Métrica en test (15 casos, H6 semilla 42) | Objetivo | Resultado |
|---|---|---|
| F1 / AUC de clasificación | 0.85 / 0.85 | 0.990 / 0.999 |
| IoU medio / mAP50 / mAP50:95 de detección | 0.65 / 0.65 / 0.40 | 0.943 / 0.961 / 0.902 |
| Dice / IoU de segmentación semántica | 0.85 / 0.70 | 0.971 / 0.943 |
| Dice / IoU por fragmento | 0.85 / 0.70 | 0.875 / 0.816 |

Las tres semillas (42, 43 y 44) dan 0.875, 0.875 y 0.876 de Dice por fragmento.

Estas cifras son una **segunda evaluación en test, posterior al análisis de fallos**; la primera
evaluación oficial fue D5 (Dice por fragmento 0.845). Cómo se eligió H6, las ablaciones, la
robustez y las limitaciones están en el [model card](MODEL_CARD.md).

## Documentación

| Documento | Contenido |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Modelo, datos, métricas, ablaciones, protocolo de test y limitaciones |
| [`docs/mini_paper/Mini_paper_PENGWIN.pdf`](docs/mini_paper/Mini_paper_PENGWIN.pdf) | Mini-paper de 5 páginas: método, resultados y limitaciones (fuente en `.docx` y `build.js`) |
| [`docs/informe/Informe_tecnico_PENGWIN.docx`](docs/informe/Informe_tecnico_PENGWIN.docx) | Informe técnico completo |
| [`IA_USAGE.md`](IA_USAGE.md) | Registro y análisis crítico del uso de IA generativa |
| [`docs/resumen_semana8.md`](docs/resumen_semana8.md) | Dataset, preprocesamiento, EDA y Visualizador 1 |
| [`docs/resumen_semana9.md`](docs/resumen_semana9.md) | Backbone, CBAM, detección y overfit intencional |
| [`docs/decisiones_semana10.md`](docs/decisiones_semana10.md) | Semana 10: bitácora de experimentos, de la línea base a H6, y las dos evaluaciones en test |
| [`outputs/s10/resumen_experimentos.md`](outputs/s10/resumen_experimentos.md) | Semana 10: tabla de todas las corridas y dónde está cada archivo de resultados |
| [`docs/semana11_dashboard.md`](docs/semana11_dashboard.md) | Semana 11: dashboard con los tres visualizadores, orientación y distancia |
| [`docs/despliegue_aws_streamlit.md`](docs/despliegue_aws_streamlit.md) | Semana 11: despliegue en Streamlit Cloud y AWS Lambda |
| [`docs/cicd_github_actions.md`](docs/cicd_github_actions.md) | Semana 11: pruebas y despliegue automáticos |
| `notebooks/01` a `05` | Seguimiento por semana: EDA, intensidades, Semana 9, Semana 10 y casos difíciles (el 05 se hizo con D5) |

Cada tema está en un solo documento. Los resultados del modelo están en el model card y no se
repiten en los demás.

## Estructura del repositorio

```
src/        scripts y módulos (ver los comandos abajo)
tests/      pruebas automáticas
notebooks/  notebooks de seguimiento por semana
kaggle/     kernels de entrenamiento en GPU (uno por receta)
colab/      genera el cuaderno de Colab que entrena H6 (notebooks/colab/colab_s10.ipynb)
app/        dashboard Streamlit
api/        API de inferencia
deploy/     scripts de despliegue
data/       manifest, splits y CSVs del EDA (data/raw/ y data/cache/ NO se versionan)
outputs/    figuras (figures/), EDA (eda/), resultados y checkpoints de la Semana 10 (s10/)
            y casos exportados para el dashboard (demo/)
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

## 3. Modelo final H6 (Semana 10)

### 3.1 Caché por caso

Guarda cada corte de cada caso a 384 px (ventana ósea + gamma) en `data/cache/s10/`:

```bash
python src/build_case_cache.py --sizes 384 --modes stretch --workers 4
```

Deja `data/cache/s10/cache_384/<caso>.npz`, `labels_native/` e `index.csv`.

### 3.2 Entrenamiento

Requiere GPU. Las tres semillas de H6 se entrenaron en Colab con una L4
(`notebooks/colab/colab_s10.ipynb` lleva esta misma receta); las recetas anteriores, en Kaggle
(`kaggle/gpu/`, un kernel por receta, incluido el de H6):

```bash
python src/train_s10.py --cache-root data/cache/s10 \
    --size 384 --epochs 60 --base-channels 32 --cbam 1 --secondary-weight 3 \
    --augment --cosine --batch-size 16 --per-epoch 2500 --max-minutes 170 --eval-every 10 \
    --ema 0.999 --seed 42 --aug-geo 1.0 --aug-rot-deg 35 --canonical \
    --output outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth
```

`--canonical` endereza los 34 volúmenes guardados en orientación LPI (ver `src/orientacion.py`).
Queda registrado en el checkpoint, así que la evaluación lo aplica sola. Las semillas 43 y 44 se
obtienen cambiando `--seed` y el nombre de salida.

El checkpoint entregado ya está en el repo, así que este paso se puede saltar.

### 3.3 Evaluación sobre el volumen completo

```bash
python src/evaluate_volumes.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta \
    --output outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42_val.json
```

Para el post-proceso 3D y la distancia de separación en mm por fragmento (también sobre el ground
truth):

```bash
python src/infer_volume.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta --distance-nofill --surface-metrics
```

H6 espera los volúmenes en orientación canónica: `evaluate_volumes.py` e `infer_volume.py` los
enderezan solos según lo que indica el checkpoint.

Resultado esperado en validación: Dice por fragmento 0.869 (puede variar en la tercera cifra según
el hardware). Reporta la detección cruda (cabeza grid) y la refinada, y separa el Dice en
fragmento primario y secundario.

**Test:** el protocolo es evaluar una sola vez con la configuración congelada. La configuración y el
comando exactos de la evaluación entregada están en `docs/decisiones_semana10.md`, sección 12.4, y
sus resultados en `outputs/s10/TEST2_h6_*.json`. No se recomienda volver a correr test para ajustar
nada.

### 3.4 Línea base SAM (solo comparación)

SAM ViT-B no forma parte del sistema: recibe como prompt la caja que predice nuestro detector y se
compara con nuestra máscara semántica. Pesos: `sam_vit_b_01ec64.pth`, del repositorio oficial de
Segment Anything.

```bash
python src/sam_baseline.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --sam-checkpoint <ruta>/sam_vit_b_01ec64.pth --cache-dir data/cache/s10/cache_384 \
    --split test --stride 3 --output outputs/s10/sam_baseline_test_h6.json
```

El enunciado pide la comparación sobre test, por eso `--split test` (sin esa opción usa validación).

### 3.5 Pruebas automáticas

```bash
python -m unittest discover -s tests -v
```

## 4. Demo y visualizadores (Semana 11)

`demo_export.py` corre el pipeline completo una vez por caso (de 4 a 25 minutos en CPU con TTA,
según la máquina y el número de cortes) y deja todo en `outputs/demo/<modelo>/<caso>/`. El
dashboard solo lee esos archivos. Casos versionados, todos con H6 y TTA: 024 (validación), 044 y 099
(validación, guardados girados 180°) y 039 (test).

```bash
python src/demo_export.py --case 024 044 --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth --tta
streamlit run app/dashboard.py
```

| Visualizador | Qué muestra |
|---|---|
| 1. Volumen crudo | MIP axial, coronal y sagital del hueso (umbral de 250 HU, sin camilla ni cables) e isosuperficie 3D. Sin modelo |
| 2. Corte por corte | Cajas, máscaras por hueso, ground truth opcional e ID y distancia en mm de cada fragmento |
| 3. Reconstrucción 3D | Una malla por fragmento, con su ID y su distancia en mm al principal |

Con H6 la demo endereza los volúmenes guardados girados 180° (cabecera `TransformMatrix = -1 0 0 0
-1 0 0 0 1`) antes de predecir, una sola vez y para las tres vistas y el ground truth; el dashboard
indica en la barra lateral si el caso se enderezó. Detalle en
[`docs/semana11_dashboard.md`](docs/semana11_dashboard.md).

### Latencia por imagen

```bash
python src/measure_latency.py --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth \
    --volume outputs/demo/h6_auggeo_rot35_canon_ema_w32_e60_s42/024/volume.npz \
    --device cpu --device cuda
```

Mide un corte a la vez y reporta la mediana y el percentil 90. Escribe en
`outputs/demo/latency.json`, que el dashboard muestra. Si no hay GPU, omite `cuda` y lo avisa. Los
resultados están en el [model card](MODEL_CARD.md), sección 4.8.

### Despliegue

El dashboard está publicado en **Streamlit Community Cloud** (app `app/dashboard.py`, rama `main`) y
la inferencia en vivo corre en una **API en AWS Lambda** con H6. Cada push a `main` corre las
pruebas y vuelve a desplegar los dos (GitHub Actions para la API, con OIDC y sin claves guardadas;
Streamlit Cloud para el dashboard). Detalle en
[`docs/despliegue_aws_streamlit.md`](docs/despliegue_aws_streamlit.md) y
[`docs/cicd_github_actions.md`](docs/cicd_github_actions.md).

## Verificación de reproducibilidad

- Con la semilla 42, `data/splits.json` sale idéntico al versionado, y `data/manifest.csv` reporta
  100 de 100 casos válidos.
- La evaluación de la sección 3.3 sobre el checkpoint entregado debe reproducir el JSON de
  validación versionado en `outputs/s10/`.

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
