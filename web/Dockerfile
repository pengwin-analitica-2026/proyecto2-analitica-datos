# Lambda pelvis-web: la API de la página de AWS (web/servidor.py) y los trabajos que procesan un CT completo.
# Se construye desde la RAÍZ del repo:  docker build --platform linux/amd64 -f web/Dockerfile -t pelvis-web .
# Misma base que api/Dockerfile; la Lambda pelvis-api y el dashboard de Streamlit no cambian.
FROM public.ecr.aws/docker/library/python:3.11-slim

# Lambda Web Adapter: sirve FastAPI tal cual y entrega las invocaciones asíncronas (los trabajos) en POST /events.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/requirements.txt web/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r web/requirements.txt

COPY src/ src/
COPY api/__init__.py api/server.py api/
COPY web/__init__.py web/servidor.py web/
COPY data/splits.json data/splits.json
COPY outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth
COPY outputs/s10/calibracion_int8_h8.npz outputs/s10/calibracion_int8_h8.npz
RUN python src/precision.py exportar --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth
RUN chmod -R a+rX /app

# HOME y temporales en /tmp (lo único escribible en Lambda). 6 hilos: lo que da Lambda con 10 GB.
ENV PORT=8080 AWS_LWA_PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/health AWS_LWA_PASS_THROUGH_PATH=/events \
    AWS_LWA_INVOKE_MODE=buffered MODEL_PATH=/app/outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \
    OMP_NUM_THREADS=6 PRECISION=auto HOME=/tmp MPLCONFIGDIR=/tmp
EXPOSE 8080
CMD ["uvicorn", "web.servidor:app", "--host", "0.0.0.0", "--port", "8080"]
