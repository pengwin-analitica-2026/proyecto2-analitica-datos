# Imagen de la API de inferencia. Sirve en AWS Lambda (con Lambda Web Adapter), App Runner o EC2.
# Se construye desde la RAÍZ del repo:  docker build --platform linux/amd64 -f api/Dockerfile -t pelvis-api .
FROM public.ecr.aws/docker/library/python:3.11-slim

# Adaptador: permite que Lambda ejecute este servidor HTTP tal cual (en EC2/App Runner no hace nada).
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r api/requirements.txt

COPY src/ src/
COPY api/server.py api/server.py
COPY outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth
# Lambda ejecuta con un usuario sin privilegios: los archivos deben ser legibles por todos
RUN chmod -R a+rX /app

ENV PORT=8080 AWS_LWA_PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/health \
    MODEL_PATH=/app/outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth OMP_NUM_THREADS=2
EXPOSE 8080
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080"]
