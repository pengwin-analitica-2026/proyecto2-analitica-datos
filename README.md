# PENGWIN: detección, segmentación y medición de fracturas pélvicas en CT

**Integrantes**

- Juan Jacobo Delgado Melo
- Federico Gonzales Cárdenas
- Gustavo Adolfo Chipantiza
- María de los Ángeles Valencia Hurtado

Proyecto Integrador Corte 2 · Analítica de Datos · Universidad Autónoma de Occidente (2026-2) ·
Profesor: Carlos Ferro.

> **Uso exclusivamente académico.** Este sistema no es un dispositivo médico, no ha sido validado
> clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Acceso al dashboard

| Versión | Enlace |
|---|---|
| App en Streamlit | https://pengwin-with-own-architecture-by-the-best-team.streamlit.app/ |
| Página web hecha desde cero en AWS (CloudFront + Lambda de 10 GB, para CT grandes) | https://d2794ruz2jvuur.cloudfront.net/ |

Funcionan en computador y en celular.

### Videos de demostración

| Video | Qué muestra | Enlace |
|---|---|---|
| Celular, caso 044 | El proyecto desde un dispositivo móvil | [Ver en YouTube](https://youtu.be/skbnub3kb8U?si=w54jJcCzoEskjVmJ) |
| Computador, caso 099 | El pipeline de detección, segmentación y clasificación | [Ver en YouTube](https://youtu.be/0ZSQLtcMnmI) |

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

## Ejecutar el dashboard en local

Requiere Python 3.11 o 3.12. El checkpoint de H8 y cuatro casos ya procesados (024, 044, 099 y 039)
vienen en el repositorio, así que el dashboard abre sin descargar datos.

### Paso 1. Clonar y crear el entorno

```bash
git clone https://github.com/pengwin-analitica-2026/proyecto2-analitica-datos.git
cd proyecto2-analitica-datos
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Paso 2A. Instalar para CPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt boto3
```

### Paso 2B. Instalar para GPU (NVIDIA)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt boto3
python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
```

### Paso 3. Abrir el dashboard

```bash
streamlit run app/dashboard.py
```

Se abre en `http://localhost:8501`.

### Paso 4. Elegir dónde corre el modelo

En la barra lateral, bloque **Procesamiento**:

- **CPU:** «Dónde corre el modelo» → **Este equipo**. En Windows elige la precisión **H8 normal
  (fp32)**, porque el PyTorch de Windows no trae el motor int8. En Linux o macOS, **Automática**
  usa int8, que es dos veces más rápido.
- **GPU:** con la instalación del paso 2B, **Este equipo** usa la GPU sola. **Automática** elige
  fp16.

Los casos ya procesados se ven al instante. «Procesar ahora» y «Subir un CT propio» (`.mha`, NIfTI, NRRD o
carpeta DICOM) corren el modelo en el destino elegido.

### Opcional: usar la CPU y la GPU de AWS desde el dashboard local

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

## Reproducir el modelo

Los CT se descargan de [Zenodo](https://doi.org/10.5281/zenodo.10927452) en `data/raw/` (no se
versionan). Todos los comandos van desde la raíz del repositorio.

```bash
python src/build_manifest.py && python src/make_splits.py                  # manifest y splits (semilla 42)
python src/build_case_cache.py --sizes 384 --modes stretch --workers 4      # caché de cortes a 384 px

# Entrenamiento de H8 (GPU; en Kaggle con una T4 tardó 67 minutos)
python src/train_s10.py --cache-root data/cache/s10 --size 384 --mode stretch --epochs 60 \
    --batch-size 16 --lr 0.002 --base-channels 32 --cbam 1 --instance-mode primary --augment --cosine \
    --margin 10 --neg-fraction 0.15 --per-epoch 2500 --val-stride 6 --eval-every 10 --max-minutes 170 \
    --secondary-weight 3.0 --canonical --aug-geo 1.0 --aug-rot-deg 35 --aug-mm --ema 0 \
    --aug-noise 0.1 --seed 42 --output outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth

# Evaluación en validación (Dice por fragmento esperado: 0.865)
python src/evaluate_volumes.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    --cache-dir data/cache/s10/cache_384 --split val --tta --output outputs/h8/validacion/val_h8_base.json

# Pruebas automáticas
python -m unittest discover -s tests -v
```

El checkpoint entregado ya está en el repositorio, así que el entrenamiento se puede saltar. Los
comandos de robustez, HD95 y ASSD, distancias, SAM, latencia y la exportación de casos para el
dashboard están en el [model card](MODEL_CARD.md) y en [`docs/modelo_h8.md`](docs/modelo_h8.md).
Test se evaluó una sola vez con la configuración congelada en
`outputs/h8/test/CONFIG_CONGELADA_TEST_H8.md`.
