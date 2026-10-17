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
