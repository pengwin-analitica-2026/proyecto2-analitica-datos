# Despliegue: API de inferencia en AWS + dashboard en Streamlit Community Cloud

Estado al 5 de octubre de 2026. Hugging Face Spaces quedó descartado: en cuentas gratis el SDK
Docker y CPU Basic están como "Paid".

```
Celular / computador ──► Streamlit Community Cloud (dashboard, datos exportados, sin torch)
                                   │  solo al pulsar «Inferir este corte en vivo»
                                   ▼
                         AWS Lambda (contenedor con el modelo, api/server.py)
```

Las tres pestañas leen los casos ya exportados (`outputs/demo/…`), que viajan con el dashboard.
La API solo atiende el botón de inferencia en vivo del visualizador 2 y devuelve la latencia del
modelo. Si la API no está configurada o está caída, el resto del dashboard sigue funcionando.

## 1. API en AWS (Lambda con contenedor)

Requisitos en el Mac: Docker Desktop y AWS CLI con credenciales (`aws configure`). La imagen se
construye desde la raíz del repo y necesita `outputs/s10/d5_ema_w32_e60_s42.pth` en disco.

```bash
export AWS_REGION=us-east-1
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REPO=pelvis-api

aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

# --provenance=false: Lambda rechaza las imágenes con atestaciones de buildx
docker build --platform linux/amd64 --provenance=false -f api/Dockerfile -t $REPO .
docker tag $REPO:latest $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

Crear la función (consola de AWS o CLI). Valores recomendados:

- Tipo: imagen de contenedor, la que se acaba de subir.
- Memoria 3008 MB (da algo más de un vCPU), tiempo máximo 60 s.
- Variable de entorno `API_KEY` con una clave larga y aleatoria (`openssl rand -hex 24`).
- URL de función con tipo de autenticación `NONE`. La clave `X-API-Key` es lo que protege la API.

Comprobación:

```bash
curl https://<tu-url>.lambda-url.us-east-1.on.aws/health
```

La primera llamada después de un rato sin uso es lenta (arranque en frío: carga torch y el modelo).
Antes de la demo conviene llamar a `/health` una vez para dejarla caliente.

Alternativa sin Lambda: la misma imagen corre en una instancia EC2 pequeña
(`docker run -p 8080:8080 -e API_KEY=… pelvis-api`), con HTTPS aparte. Es más simple de entender pero
hay que apagarla para que no cobre.

## 2. Dashboard en Streamlit Community Cloud

```bash
python3 deploy/build_streamlit_cloud.py --model d5_ema_w32_e60_s42 --cases 024 099 044
```

Genera `build/streamlit_cloud/` (≈65 MB, el archivo más grande pesa 28 MB, bajo el límite de 100 MB
de GitHub). Esa carpeta es un repositorio aparte:

```bash
cd build/streamlit_cloud
git init && git add . && git commit -m "Dashboard Semana 11"
# crear un repo vacío en GitHub (privado sirve) y:
git remote add origin git@github.com:<usuario>/<repo>.git
git push -u origin main
```

En share.streamlit.io: New app → ese repo, rama `main`, archivo principal `app/dashboard.py`.
En Advanced settings: Python 3.11 y, en Secrets:

```toml
INFERENCE_URL = "https://<tu-url>.lambda-url.us-east-1.on.aws"
INFERENCE_KEY = "<la misma clave que API_KEY>"
```

Sin esos secretos el expander de inferencia en vivo avisa que no está configurado y todo lo demás
funciona.

## Pruebas hechas (en el entorno de desarrollo, no en AWS ni en Streamlit Cloud)

- La API local responde 401 con clave incorrecta, 400 con un cuerpo que no es `.npy` y 200 con un
  corte válido (≈190 ms de modelo en CPU de 2 núcleos).
- El dashboard armado en `build/streamlit_cloud/`, con `import torch` bloqueado a propósito,
  arranca, muestra las tres pestañas y completa la inferencia en vivo contra la API.

## Sin verificar

- Que Lambda y la imagen funcionen tal cual (no hay AWS en el entorno donde se desarrolló).
- La memoria disponible en Streamlit Community Cloud con los casos cargados.
- Costos actuales de AWS: con el uso de una demo deberían ser mínimos, pero conviene mirar la
  consola de facturación.
- La latencia de Lambda: es la que hay que reportar como "CPU en la nube", medida con la API ya
  desplegada.
