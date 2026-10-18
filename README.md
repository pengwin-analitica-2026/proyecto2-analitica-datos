# PENGWIN: detección, segmentación y medición de fracturas pélvicas en CT

**Integrantes**

| Integrante | Código |
|---|---|
| Gustavo Adolfo Chipantiza | 2222022 |
| Federico Gonzales Cárdenas | 2225657 |
| Juan Jacobo Delgado Melo | 2230110 |
| María de los Ángeles Valencia Hurtado | 2235112 |

Proyecto Integrador Corte 2 · Analítica de Datos · Universidad Autónoma de Occidente (2026-2) ·
Profesor: Carlos Ferro.

> **Uso exclusivamente académico.** Este sistema no es un dispositivo médico, no ha sido validado
> clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Enlaces

| Qué | Enlace |
|---|---|
| App en Streamlit | https://pengwin-with-own-architecture-by-the-best-team.streamlit.app/ |
| Página web hecha desde cero en AWS | https://d2794ruz2jvuur.cloudfront.net/ |
| Video celular, caso 044 | [Ver en YouTube](https://youtu.be/skbnub3kb8U?si=w54jJcCzoEskjVmJ) |
| Video computador, caso 099 | [Ver en YouTube](https://youtu.be/0ZSQLtcMnmI) |
| Paper | [Google Docs](https://docs.google.com/document/d/1-XUDOoewhVg3ZzZrI3hx3H6rg2GX0qRyfQAbdp5Y42k/edit?usp=sharing) · PDF en [`docs/mini_paper/Mini_paper_PENGWIN.pdf`](docs/mini_paper/Mini_paper_PENGWIN.pdf) |
| Repositorio | https://github.com/pengwin-analitica-2026/proyecto2-analitica-datos |

### ¿Cuál de las dos versiones usar?

Las dos hacen lo mismo y se pueden usar indistintamente, en computador o en celular: casos de
ejemplo, los 100 casos de PENGWIN, subir un CT propio, los tres visualizadores, la inferencia en vivo
de un corte y las métricas.

| | App en Streamlit | Página web en AWS |
|---|---|---|
| Dónde procesa | En el mismo servidor de Streamlit Cloud (~2.7 GB de memoria) | En una Lambda propia de AWS con 10 GB y 15 minutos; el navegador solo dibuja |
| Para qué conviene | Explorar los casos de ejemplo y los precalculados | **Lo pesado**: subir un CT grande (como el `.mha` del caso 044), una carpeta de cortes o «Procesar ahora» |
| Código | [`app/dashboard.py`](app/dashboard.py) | [`web/`](web/) (React + three.js y la API en FastAPI) |

Si se va a subir un volumen grande o procesar un caso completo en vivo, mejor la página de AWS:
Streamlit Cloud se puede quedar sin memoria con esos volúmenes.

## Contexto

Para planear la cirugía de una fractura de pelvis hay que saber cuántos fragmentos hay, a qué hueso
pertenece cada uno y cuánto se desplazó. Con los 100 CT del reto
[PENGWIN 2024](https://pengwin.grand-challenge.org/) (`.mha`, CC BY 4.0, partición por paciente
70 / 15 / 15 con semilla 42), el sistema hace en cada corte cuatro tareas:

1. **Clasifica** qué regiones óseas aparecen (sacro, coxal izquierdo, coxal derecho).
2. **Detecta** cada región con una caja (grid propio y supresión de no máximos escrita por el equipo).
3. **Segmenta** cada región y separa el fragmento principal de los secundarios.
4. **Mide** en milímetros la distancia borde a borde de cada fragmento al principal de su hueso, con
   el espaciado real de la cabecera.

Todo con una red 2D propia, sin YOLO, Detectron2, Mask R-CNN ni frameworks de segmentación.
SAM se usa solo como comparación.

## Cómo se resolvió

- **Modelo propio.** `FundidoraBackbone` (extensión de FundidoraPC, 32 canales) con atención CBAM
  antes de bifurcar en tres cabezas: clasificación, detección y segmentación. 2.74 M de parámetros,
  cortes de 384 × 384 con ventana ósea y gamma 1.5.
- **Orientación.** Un tercio de los volúmenes venía guardado girado 180°. Se corrige con productos
  punto y cruz sobre la matriz de dirección de la cabecera, y la demo endereza además el giro del
  paciente en la camilla.
- **Modelo final H8.** Sin EMA, giro del aumento definido en milímetros (±35°) y aumento con ruido
  gaussiano. Con ruido fuerte conserva 0.835 de Dice por fragmento, donde los modelos anteriores
  caían por debajo de 0.23.
- **Volumen 3D.** Los cortes se apilan, un post-proceso 3D une los fragmentos y la distancia se mide
  con la transformada de distancia euclídea.
- **Despliegue.** Dashboard en Streamlit Cloud y página web en AWS, con la inferencia en AWS Lambda
  (int8 en CPU) y una GPU bajo demanda en SageMaker (fp16). Cada push a `main` corre las pruebas y
  vuelve a desplegar.

## Resultados (H8, semilla 42, 15 casos por partición)

| Métrica | Objetivo | Validación | Test |
|---|---|---|---|
| F1 / AUC de clasificación | 0.85 / 0.85 | 0.991 / 0.999 | 0.989 / 0.999 |
| IoU / mAP50 / mAP50:95 de detección | 0.65 / 0.65 / 0.40 | 0.946 / 0.969 / 0.900 | 0.938 / 0.953 / 0.898 |
| Dice / IoU de segmentación semántica | 0.85 / 0.70 | 0.970 / 0.943 | 0.968 / 0.939 |
| Dice / IoU por fragmento | 0.85 / 0.70 | 0.865 / 0.800 | 0.874 / 0.812 |

Cumple los nueve objetivos en validación y en test, y con tres semillas el Dice por fragmento en
test es 0.875 ± 0.001. Además:

- HD95 y ASSD por fragmento, medianas de test: 2.73 y 0.66 mm.
- Error medio de la distancia en test: 1.52 mm.
- SAM con nuestras cajas: 0.889 de Dice, frente a 0.931 de H8.
- Cuantización sin pérdida de Dice: int8 en CPU (81 frente a 183 ms por corte) y fp16 en GPU.

## Documentos

| Documento | Contenido |
|---|---|
| [`docs/mini_paper/Mini_paper_PENGWIN.pdf`](docs/mini_paper/Mini_paper_PENGWIN.pdf) | Mini-paper (4 páginas), con su fuente en `.docx` |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Model card: arquitectura, datos, métricas, ablaciones, protocolo de test, limitaciones y advertencia de uso |
| [`IA_USAGE.md`](IA_USAGE.md) | Registro y análisis crítico del uso de IA generativa |
| [`docs/modelo_h8.md`](docs/modelo_h8.md) | Modelo final H8: qué cambia frente a H6 y resultados |
| [`docs/experimentos_ema_rotacion_ruido.md`](docs/experimentos_ema_rotacion_ruido.md) | Comparación de los cinco modelos después de la asesoría |
| [`docs/cuantizacion.md`](docs/cuantizacion.md) | Cuantización int8 y fp16 |
| [`docs/resumen_semana8.md`](docs/resumen_semana8.md), [`docs/resumen_semana9.md`](docs/resumen_semana9.md), [`docs/decisiones_semana10.md`](docs/decisiones_semana10.md), [`docs/semana11_dashboard.md`](docs/semana11_dashboard.md) | Avance por semana |
| [`docs/despliegue_aws_streamlit.md`](docs/despliegue_aws_streamlit.md), [`docs/web_aws.md`](docs/web_aws.md), [`docs/cicd_github_actions.md`](docs/cicd_github_actions.md) | Despliegue y CI/CD |
| `notebooks/` | Seguimiento por semana (01 a 07) y el modelo final (`11_modelo_h8.ipynb`) |

## Ejecutar las aplicaciones

### A. App de Streamlit en local

Requiere Python 3.11 o 3.12. El checkpoint de H8 y cuatro casos ya procesados (024, 044, 099 y 039)
vienen en el repositorio, así que el dashboard abre sin descargar datos.

#### Paso 1. Clonar y crear el entorno

```bash
git clone https://github.com/pengwin-analitica-2026/proyecto2-analitica-datos.git
cd proyecto2-analitica-datos
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

#### Paso 2A. Instalar para CPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt boto3
```

#### Paso 2B. Instalar para GPU (NVIDIA)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt boto3
python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
```

#### Paso 3. Abrir el dashboard

```bash
streamlit run app/dashboard.py
```

Se abre en `http://localhost:8501`.

#### Paso 4. Elegir dónde corre el modelo

En la barra lateral, bloque **Procesamiento**:

- **CPU:** «Dónde corre el modelo» → **Este equipo**. En Windows elige la precisión **H8 normal
  (fp32)**, porque el PyTorch de Windows no trae el motor int8. En Linux o macOS, **Automática**
  usa int8, que es dos veces más rápido.
- **GPU:** con la instalación del paso 2B, **Este equipo** usa la GPU sola. **Automática** elige
  fp16.

Los casos ya procesados se ven al instante. «Procesar ahora» y «Subir un CT propio» corren el modelo en el destino elegido. Se puede
subir un volumen en un archivo (`.mha`, NIfTI o NRRD) o una carpeta de cortes: una serie DICOM o un
`.mha` por corte. Si los `.mha` son 2D, se indica el grosor entre cortes en mm.

#### Opcional: usar la CPU y la GPU de AWS desde el dashboard local

1. Crea `.streamlit/secrets.toml` con la dirección y la clave de la API (pídela al equipo):

   ```toml
   INFERENCE_URL = "https://xr3mjj5zdgwq2q4jfgxjddad7a0dljsj.lambda-url.us-east-2.on.aws"
   INFERENCE_KEY = "<clave de la API>"
   ```

2. Reinicia el dashboard. En **Procesamiento** aparece **AWS · CPU (Lambda)**, y con eso también se
   pueden abrir los 100 casos de PENGWIN.
3. Para la GPU, en el panel **GPU (SageMaker)** pulsa **Encender (4 h)** con la clave de
   administración. Tarda unos 8 minutos. Cuando está encendida aparece **AWS · GPU (SageMaker)**.
4. Al terminar pulsa **Apagar**. Si nadie la apaga, se apaga sola a las 4 horas.

### B. Página web de AWS en local

La misma API que corre en la Lambda (`web/servidor.py`) se puede correr en el equipo: guarda todo en
una carpeta, procesa los trabajos en un hilo y sirve la página ya compilada. Usa el entorno del paso A
y además Node 20 o más reciente.

```bash
pip install -r web/requirements.txt                       # FastAPI, uvicorn y python-multipart
cd web/frontend && npm ci && npm run build && cd ../..    # compila la página en web/frontend/dist

# Casos de ejemplo y métricas para la página (salen de outputs/demo/, que viene en el repositorio)
python src/paquete_web.py outputs/demo/h8_aumm_noise01_noema_w32_e60_s42/* \
    --destino datos_web/resultados/h8_aumm_noise01_noema_w32_e60_s42/ejemplos \
    --resumen datos_web/datos/resumen.json

WEB_DATA_DIR=datos_web uvicorn web.servidor:app --port 8080
```

Se abre en `http://localhost:8080`. Los casos de ejemplo y la subida de un CT funcionan sin AWS. Para
listar los 100 casos de PENGWIN hacen falta credenciales con lectura del bucket (en el equipo,
`AWS_PROFILE=almia` antes de `uvicorn`). Para trabajar en la interfaz con recarga en vivo, deja la API
corriendo y en otra terminal usa `cd web/frontend && npm run dev` (Vite en `http://localhost:5173`).

### C. Desplegar en AWS

**Página web** (S3 + CloudFront, y la Lambda `pelvis-web` de 10 GB). Requiere AWS CLI con el perfil de
la cuenta, Docker y Node:

```bash
AWS_PROFILE=almia bash deploy/aws/web.sh todo      # la primera vez: bucket, rol, imagen, Lambda, CloudFront y página
AWS_PROFILE=almia bash deploy/aws/web.sh imagen    # después de cambiar web/servidor.py o src/
AWS_PROFILE=almia bash deploy/aws/web.sh pagina    # después de cambiar web/frontend/
AWS_PROFILE=almia bash deploy/aws/web.sh estado    # direcciones y configuración
```

**App de Streamlit.** Está en Streamlit Community Cloud (archivo `app/dashboard.py`, rama `main`) y se
vuelve a desplegar sola con cada push a `main`. La API de inferencia `pelvis-api` (AWS Lambda) también:
GitHub Actions corre las pruebas, construye la imagen y actualiza la función.

Detalle en [`docs/web_aws.md`](docs/web_aws.md), [`docs/despliegue_aws_streamlit.md`](docs/despliegue_aws_streamlit.md)
y [`docs/cicd_github_actions.md`](docs/cicd_github_actions.md).

## Reproducir el modelo

Los CT se descargan de [Zenodo](https://doi.org/10.5281/zenodo.10927452) (CC BY 4.0) y no se versionan.
Se descomprimen con esta estructura exacta:

```
data/raw/
├── PENGWIN_CT_train_images_part1/   001.mha …
├── PENGWIN_CT_train_images_part2/   … 100.mha
└── PENGWIN_CT_train_labels/         001.mha … 100.mha
```

Todos los comandos van desde la raíz del repositorio, con el entorno activo. El checkpoint de H8 ya
está en el repositorio, así que el entrenamiento (3.2) se puede saltar.

### 1. Datos y EDA (Semana 8)

El paso 1 va primero: los pasos 2, 3, 6 y 7 leen `data/manifest.csv`.

```bash
python src/build_manifest.py                      # 1. data/manifest.csv
python src/check_hu_outliers.py                   # 2. diagnóstico de HU atípicos
python src/make_splits.py                         # 3. data/splits.json (70/15/15, semilla 42)
python src/hu_windowing_check.py                  # 4. outputs/figures/hu_window_check_*.png
python src/volumen_3d_visualizer.py               # 5. visualizador 3D: outputs/figures/volumen_3d_<id>.html
python src/eda_fragments.py                       # 6. data/eda_fragments_detalle.csv + outputs/eda/
python src/eda_fragmento_principal_y_tamano.py    # 7. data/eda_fragmentos_individuales.csv + outputs/eda/
```

### 2. Detección (Semana 9)

```bash
python src/train_overfit_detection.py --case-id 001 --epochs 150   # overfit intencional
python src/visualize_overfit_detection.py --case-id 001            # cajas GT contra predicción
python src/prepare_week9_detection_data.py                          # caché de cortes train/val
python src/train_week9_detection.py                                 # primer entrenamiento multi-paciente
python src/visualize_week9_validation.py --case-id 014              # validación cualitativa
```

### 3. Modelo final H8

```bash
# 3.1 Caché de cortes a 384 px (ventana ósea + gamma): data/cache/s10/cache_384/
python src/build_case_cache.py --sizes 384 --modes stretch --workers 4

# 3.2 Entrenamiento (GPU; en Kaggle con una T4 tardó 67 minutos; los kernels están en kaggle/h8/)
python src/train_s10.py --cache-root data/cache/s10 --size 384 --mode stretch --epochs 60 \
    --batch-size 16 --lr 0.002 --base-channels 32 --cbam 1 --instance-mode primary --augment --cosine \
    --margin 10 --neg-fraction 0.15 --per-epoch 2500 --val-stride 6 --eval-every 10 --max-minutes 170 \
    --secondary-weight 3.0 --canonical --aug-geo 1.0 --aug-rot-deg 35 --aug-mm --ema 0 \
    --aug-noise 0.1 --seed 42 --output outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth

# 3.3 Evaluación en validación (Dice por fragmento esperado: 0.865)
python src/evaluate_volumes.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta --output outputs/h8/validacion/val_h8_base.json

# Robustez (solo validación): agregar --stress noise=10, --stress noise=25, --stress rotmm=60
# o --stress rotmm=60 --normalizar al comando anterior

# HD95 y ASSD por fragmento
python src/metricas_superficie.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta \
    --output outputs/h8/validacion/superficie_val_h8.json

# Distancia de separación en mm, sobre la predicción y sobre el ground truth
python src/infer_volume.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta --distance-nofill --surface-metrics \
    --output outputs/h8/validacion/distancias_val_h8.json

# 3.4 Línea base SAM ViT-B (solo comparación; pesos sam_vit_b_01ec64.pth del repositorio oficial)
python src/sam_baseline.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --sam-checkpoint <ruta>/sam_vit_b_01ec64.pth --cache-dir data/cache/s10/cache_384 \
    --split test --stride 3 --output outputs/h8/test/sam_baseline_test_h8.json

# 3.5 Pruebas automáticas
python -m unittest discover -s tests -v
```

Test se evaluó una sola vez con la configuración congelada en
`outputs/h8/test/CONFIG_CONGELADA_TEST_H8.md`; sus resultados están en `outputs/h8/test/`. No se
recomienda volver a correr test para ajustar nada.

### 4. Casos para las aplicaciones y latencia (Semana 11)

`demo_export.py` corre el pipeline completo una vez por caso (de 4 a 25 minutos en CPU con TTA) y deja
todo en `outputs/demo/<modelo>/<caso>/`, que leen las dos aplicaciones. Los casos 024, 039, 044 y 099
ya vienen en el repositorio.

```bash
python src/demo_export.py --case 024 044 --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth --tta

# Latencia por imagen: outputs/demo/latency.json, que muestran las dos aplicaciones
python src/measure_latency.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --volume outputs/demo/h8_aumm_noise01_noema_w32_e60_s42/024/volume.npz --device cpu --device cuda
```

### Verificación de reproducibilidad

- Con la semilla 42, `data/splits.json` sale idéntico al versionado, y `data/manifest.csv` reporta
  100 de 100 casos válidos.
- La evaluación de la sección 3.3 sobre el checkpoint entregado reproduce el JSON de validación
  versionado en `outputs/h8/validacion/` (puede variar en la tercera cifra según el hardware).

## Estructura del repositorio

```
src/        scripts y módulos del pipeline
tests/      pruebas automáticas
notebooks/  seguimiento por semana y el modelo final (11_modelo_h8.ipynb)
kaggle/     kernels de entrenamiento en GPU (los de H8 en kaggle/h8/)
colab/      genera el cuaderno de Colab que entrena H6
app/        dashboard de Streamlit
web/        página web de AWS: API (servidor.py) y frontend (React + three.js)
api/        API de inferencia por corte (Lambda pelvis-api y GPU en SageMaker)
deploy/     scripts de despliegue (deploy/aws/web.sh para la página web)
data/       manifest, splits y CSV del EDA (data/raw/ y data/cache/ no se versionan)
outputs/    figuras, EDA, resultados, checkpoints y casos exportados para las aplicaciones (demo/)
docs/       un documento por semana o por tema, mini-paper e informe técnico
```

## Restricciones del proyecto

Prohibido usar YOLO, Detectron2, Mask R-CNN preentrenado o frameworks de detección o segmentación de
alto nivel. Transfer learning solo en el backbone; las cabezas se entrenan desde cero. SAM solo como
línea base zero-shot, nunca en el pipeline.
