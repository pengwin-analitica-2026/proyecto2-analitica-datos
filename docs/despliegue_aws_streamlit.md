# Despliegue: Streamlit Community Cloud y API en AWS Lambda

Estado al 13 de octubre de 2026. Modelo de la imagen de la API: **H8** (`h8_aumm_noise01_noema_w32_e60_s42.pth`).
La Lambda publicada se actualiza sola cuando esta versión llegue a `main`; hasta entonces sigue con
H6, y el dashboard avisa si el modelo del servicio no es el del caso exportado.

```
Celular / computador ──► Streamlit Community Cloud (app/dashboard.py, sin torch)
                                 │  solo al pulsar «Inferir el corte seleccionado»
                                 ▼
                       AWS Lambda pelvis-api (contenedor con H8, api/server.py)
```

Las tres pestañas leen los casos ya exportados (`outputs/demo/<modelo>/<caso>/`), que viajan en el
repositorio. El modelo solo corre en vivo al pulsar el botón del visualizador 2. Si la API no
responde, el resto del dashboard sigue funcionando.

**Diferencia con el enunciado, aprobada por el profesor.** El enunciado pide el dashboard
«desplegado en vivo mediante un túnel de Cloudflare». El profesor Carlos Ferro aprobó usar otro
medio de despliegue, y el equipo usó Streamlit Cloud y AWS Lambda: una dirección fija que se abre
desde computador y celular sin depender de que un equipo del grupo esté encendido. El túnel no se
montó. Si hiciera falta, se abre en dos comandos desde un equipo con el repositorio y `cloudflared`
instalados:

```bash
streamlit run app/dashboard.py
cloudflared tunnel --url http://localhost:8501     # en otra terminal; imprime la dirección pública
```

## 1. Dashboard en Streamlit Community Cloud

- App en el espacio de trabajo `pengwin-analitica-2026` de share.streamlit.io.
- Repositorio `pengwin-analitica-2026/proyecto2-analitica-datos`, rama `main`, archivo
  `app/dashboard.py`, Python 3.11.
- Dependencias: `app/requirements.txt` (sin torch). Streamlit lo prefiere al `requirements.txt` de la
  raíz porque está junto al archivo principal.
- Se vuelve a desplegar sola con cada push a `main`.
- Secretos (App → Settings → Secrets):

```toml
INFERENCE_URL = "https://xr3mjj5zdgwq2q4jfgxjddad7a0dljsj.lambda-url.us-east-2.on.aws"
INFERENCE_KEY = "<el valor de API_KEY de la Lambda>"
```

La clave se lee con:

```bash
AWS_PROFILE=almia aws lambda get-function-configuration --function-name pelvis-api --region us-east-2 --query 'Environment.Variables.API_KEY' --output text
```

**Primer despliegue.** Falló con «Deploy keys are disabled for this repository or organization»:
Streamlit Cloud lee los repos privados con una *deploy key* y la organización las tenía desactivadas.
Se habilitaron en la configuración de la organización y el despliegue pasó.

`deploy/build_streamlit_cloud.py` sigue sirviendo para publicar el dashboard desde un repositorio
aparte, solo con la app y los casos, si algún día el repo principal no puede conectarse.

### Casos PENGWIN desde S3

El origen «Casos PENGWIN (100)» lee el bucket privado `pengwin-ct-050752608083` (ver
`docs/semana11_dashboard.md`). **No hace falta agregar credenciales de AWS a Streamlit Cloud**: con los
mismos Secrets de siempre (`INFERENCE_URL` e `INFERENCE_KEY`), el dashboard le pide a la API la lista
de casos (`GET /s3/listar`) y un enlace firmado de 15 minutos para cada archivo (`GET /s3/url`). La
Lambda lee el bucket con su propio rol, que tiene solo lectura de `crudos/` y `exportados/`
(`deploy/aws/lambda-lectura-s3-policy.json`). Los dos endpoints exigen la `X-API-Key` y rechazan
cualquier clave fuera de esos prefijos.

`src/almacen_s3.py` decide solo: si boto3 encuentra credenciales (en el Mac, `AWS_PROFILE=almia`) va
directo al bucket; si no, pasa por la API. Opcionalmente, el usuario IAM `pengwin-streamlit-lectura`
(solo lectura, `deploy/aws/streamlit-lectura-policy.json`) permite el acceso directo si se crea su clave
y se pone en los Secrets como `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY`; no es necesario.

## 2. API de inferencia en AWS Lambda

| Recurso | Valor |
|---|---|
| Cuenta / región | Almia (`050752608083`) / `us-east-2` |
| Repositorio ECR | `pelvis-api` (escaneo de vulnerabilidades al subir) |
| Función Lambda | `pelvis-api`: imagen de contenedor, 3008 MB, tiempo máximo 120 s |
| Rol de ejecución | `pelvis-api-lambda-exec` (solo `AWSLambdaBasicExecutionRole`, para los logs) |
| URL | Function URL pública con autenticación `NONE`; `/predict_slice` y `/predict_batch` exigen la cabecera `X-API-Key` |

**Precisión.** La variable `PRECISION` (por defecto `auto`) elige cómo corre H8: en la Lambda (CPU)
**int8 híbrido**, generado al construir la imagen (`python src/precision.py exportar`, con la
calibración `outputs/s10/calibracion_int8_h8.npz`); en un servidor con GPU sería fp16; `fp32` fuerza el
modelo original. Mismo Dice en los tres (`docs/cuantizacion.md`). `/health?deep=1` informa `precision`
y `precision_origen`, y `/predict_slice` y `/predict_batch` devuelven `precision`.

### GPU bajo demanda (SageMaker)

La misma API en una imagen con CUDA (`api/Dockerfile.gpu`, torch 2.5.1 + CUDA 12.1), servida por un
endpoint de SageMaker en **`ml.g4dn.xlarge`** (NVIDIA T4, 16 GB, ~0.74 USD/hora en us-east-2). Con
`PRECISION=auto` usa fp16, 5-6 ms por corte en una T4 (medido en Kaggle). Se eligió la T4 porque H8 pesa
11 MB y con un modelo tan pequeño una L4 (`ml.g6.xlarge`, 1.13 USD/h) o una A10G (`ml.g5.xlarge`,
1.41 USD/h) casi no ganan: el tiempo lo pone la sobrecarga, no el cálculo.

**Sin credenciales nuevas en Streamlit.** Un endpoint de SageMaker no tiene URL pública. El dashboard le
pide a la Lambda con la cabecera `X-Destino: gpu`, y la Lambda (con su rol: `sagemaker:InvokeEndpoint` y
`DescribeEndpoint` sobre ese endpoint) se lo reenvía y devuelve la respuesta. `/health` informa el estado
del endpoint (`gpu`); el dashboard muestra «AWS · GPU (SageMaker)» en «Procesamiento» solo cuando está
`InService`. El contenedor atiende las rutas que exige SageMaker: `/ping` y `/invocations` (la ruta
interna va en `CustomAttributes`).

**Cobra por hora mientras el endpoint exista**, así que se enciende para probar o presentar y se apaga.
Apagar es borrar el endpoint; el modelo, la configuración, la imagen y los roles no cobran por hora y se
quedan, así que volver a encender es solo crear el endpoint otra vez (~8 minutos hasta `InService`).

```bash
AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh imagen     # GitHub Actions construye y sube la imagen (~15 min)
AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh preparar   # una vez: modelo, configuración, roles, variables de la Lambda
AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh encender   # crea el endpoint y programa el apagado a las 4 h
AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh apagar     # borra el endpoint y cancela el apagado programado
AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh borrar     # además borra el modelo y la configuración
```

**Botón en el dashboard.** En la barra lateral, «GPU (SageMaker)» muestra el estado (apagada,
encendiendo, encendida) y cuánto falta para el apagado automático, con los botones «Encender (4 h)»
(«Extender a 4 h» si ya está encendida) y «Apagar». La Lambda los atiende en `POST /gpu/encender` y
`POST /gpu/apagar` (`src/gpu_bajo_demanda.py`); `GET /gpu/estado` solo pide la `X-API-Key`. Como la app
es pública y cada encendido puede costar hasta ~3 USD, encender y apagar piden además una **clave de
administración** (`GPU_ADMIN_KEY` en la Lambda, cabecera `X-Admin-Key`); sin ella responden 403.
`preparar` la genera una vez y la deja en `~/.config/pengwin/gpu_admin_key` (permisos 600) para
compartirla con el equipo por un canal privado.

**Apagado automático a las 4 horas.** Al encender (por el botón o por el script) se programa en
EventBridge Scheduler una tarea única, `apagar-pelvis-h8-gpu`, con `at(<hora UTC>)` que llama a
`sagemaker:DeleteEndpoint` y se borra sola al ejecutarse. Encender otra vez con la GPU ya encendida
corre esa hora a 4 horas desde ese momento; apagar a mano la cancela. Así, si nadie se acuerda de
apagarla, el peor caso son 4 horas (~3 USD). El programador usa su propio rol, `pelvis-gpu-apagado`, que
solo puede borrar ese endpoint. La Lambda puede crear y borrar solo ese endpoint (desde su configuración),
manejar solo esa tarea y pasarle solo ese rol al programador. `GPU_HORAS_APAGADO` cambia las 4 horas.

La imagen (~3 GB) la construye el workflow `build-gpu.yml`, a mano: desde una conexión de casa la capa
de PyTorch con CUDA (3.1 GB) se cortaba a mitad de la subida y Docker la reintentaba desde cero. Construir
no cuesta en SageMaker. El repositorio ECR `pelvis-api-gpu` conserva las 3 imágenes más recientes.
`INFERENCE_URL_GPU` sigue disponible para apuntar a otro servidor con GPU sin pasar por la Lambda.

Endpoints: `GET /health` (al instante), `GET /health?deep=1` (carga el modelo), `POST /predict_slice`
(un corte, para la inferencia en vivo del visualizador 2) y `POST /predict_batch` (hasta 16 cortes
uint8 de 384×384 en un `.npy`; lo usa el dashboard para inferir un CT subido completo, ver
`docs/semana11_dashboard.md`). Lambda acepta hasta 6 MB por pedido; 16 cortes son ~2.4 MB.

La función se creó una vez a mano (imagen `bootstrap`); desde entonces la actualiza
`deploy-api.yml` en cada push a `main` (ver `docs/cicd_github_actions.md`).

Una Function URL pública necesita **dos** permisos en la política de la función: `lambda:InvokeFunctionUrl`
y `lambda:InvokeFunction` con la condición `lambda:InvokedViaFunctionUrl`. Con solo el primero
responde 403. El AWS CLI 2.27 no tiene la opción del segundo; se agregó con boto3 reciente.

### Orientación

H8 espera cortes en orientación canónica. La API no gira nada: el dashboard le manda cortes del
caso exportado, y `demo_export.py` ya enderezó el volumen. `/health?deep=1` devuelve
`"canonical": true` para que se pueda comprobar.

### Comprobación

```bash
URL=https://xr3mjj5zdgwq2q4jfgxjddad7a0dljsj.lambda-url.us-east-2.on.aws
curl $URL/health              # al instante: {"ok": true, "loaded": false|true}
curl "$URL/health?deep=1"     # carga el modelo; en frío tarda hasta ~90 s
```

## Qué se probó de verdad

Contra la Lambda real, el 6 de octubre:

- `/health` responde 200 en 2.8 s; `/health?deep=1` en frío tardó 93 s y devolvió el modelo,
  `cpu` y 2 736 958 parámetros.
- `/predict_slice` con un corte de 384×384: 200 con la clave y 401 sin ella o con una clave
  equivocada. Ida y vuelta ~1.1 s ya caliente; 321 ms de modelo en la CPU de Lambda.
- `deploy-api.yml` completo desde `main`: pruebas, imagen, ECR, actualización y prueba de humo.

Después de integrar la demo con H6 a `main` (6 de octubre), comprobado por Gustavo:

- La Lambda quedó redesplegada con H6 y la inferencia en vivo responde desde el dashboard.
- El dashboard de Streamlit Cloud abre y funciona desde un celular: los tres visualizadores, la
  tabla de distancias, la pestaña de métricas y la inferencia en vivo de un corte (753 ms de modelo
  en la CPU de Lambda). La prueba está grabada en `docs/video/demo_celular.mp4` (70 segundos, sin
  audio).

Sin verificar todavía:

- Costos: con el uso de una demo deberían ser mínimos (Lambda cobra por invocación y ECR por
  almacenamiento, unos 0.5 GB por imagen). Las imágenes viejas de ECR se pueden borrar.

## Al terminar el curso

Borrar el bucket `pengwin-ct-050752608083` y el usuario `pengwin-streamlit-lectura`, la función `pelvis-api`, el repositorio ECR `pelvis-api`, los roles
`github-actions-pelvis-api` y `pelvis-api-lambda-exec`, y la app de Streamlit Cloud.
