#!/usr/bin/env bash
# GPU bajo demanda para la demo: la misma API (api/server.py) en un endpoint de SageMaker con una
# NVIDIA T4 (ml.g4dn.xlarge, ~0.74 USD/hora en us-east-2). Cobra por hora MIENTRAS EXISTA: se crea para
# probar o presentar y se borra después.
#
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh imagen    # GitHub Actions construye y sube la imagen (~15 min)
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh crear     # modelo + configuración + endpoint (~8 min)
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh estado
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh borrar    # borra el endpoint, su configuración y el modelo
#
# La imagen (~3 GB) la construye GitHub Actions (.github/workflows/build-gpu.yml): desde una conexión de casa
# la subida a ECR se corta. «imagen-local» la construye y sube desde este computador.
#
# La Lambda de la API hace de puente (cabecera X-Destino: gpu): «crear» le pone SAGEMAKER_ENDPOINT y el
# permiso de invocar el endpoint. Con el endpoint borrado, el dashboard deja de ofrecer la GPU solo.
# La imagen, el repositorio ECR y los roles se conservan (no cobran por hora); «borrar --todo» también
# los quita.
set -euo pipefail
cd "$(dirname "$0")/../.."

REGION="${AWS_REGION:-us-east-2}"
CUENTA="$(aws sts get-caller-identity --query Account --output text)"
REPO=pelvis-api-gpu
ETIQUETA="${ETIQUETA:-latest}"
IMAGEN="$CUENTA.dkr.ecr.$REGION.amazonaws.com/$REPO:$ETIQUETA"
NOMBRE=pelvis-h8-gpu
INSTANCIA="${INSTANCIA:-ml.g4dn.xlarge}"
ROL=pelvis-sagemaker-exec
LAMBDA=pelvis-api
ROL_LAMBDA=pelvis-api-lambda-exec
export AWS_DEFAULT_REGION="$REGION"

imagen() {
  # Lanza el workflow de GitHub Actions y espera a que termine (requiere gh con sesión iniciada).
  local repo="${REPO_GITHUB:-pengwin-analitica-2026/proyecto2-analitica-datos}"
  gh workflow run build-gpu.yml --repo "$repo" --ref "${RAMA:-main}"
  sleep 15
  local id
  id="$(gh run list --repo "$repo" --workflow build-gpu.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
  echo "Construyendo en GitHub Actions (run $id)…"
  gh run watch "$id" --repo "$repo" --exit-status --interval 30 > /dev/null
  echo "Imagen: $IMAGEN"
}

imagen_local() {
  aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$REPO" --image-scanning-configuration scanOnPush=true >/dev/null
  aws ecr get-login-password | docker login --username AWS --password-stdin "$CUENTA.dkr.ecr.$REGION.amazonaws.com" >/dev/null
  docker build --platform linux/amd64 --provenance=false -f api/Dockerfile.gpu -t "$IMAGEN" .
  docker push "$IMAGEN"
  echo "Imagen: $IMAGEN"
}

rol_sagemaker() {
  if ! aws iam get-role --role-name "$ROL" >/dev/null 2>&1; then
    aws iam create-role --role-name "$ROL" --assume-role-policy-document \
      '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  fi
  # Solo bajar su imagen y escribir sus logs.
  aws iam put-role-policy --role-name "$ROL" --policy-name pelvis-gpu-minimo --policy-document "{
    \"Version\":\"2012-10-17\",\"Statement\":[
      {\"Effect\":\"Allow\",\"Action\":\"ecr:GetAuthorizationToken\",\"Resource\":\"*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"ecr:BatchGetImage\",\"ecr:GetDownloadUrlForLayer\",\"ecr:BatchCheckLayerAvailability\"],
       \"Resource\":\"arn:aws:ecr:$REGION:$CUENTA:repository/$REPO\"},
      {\"Effect\":\"Allow\",\"Action\":[\"logs:CreateLogGroup\",\"logs:CreateLogStream\",\"logs:PutLogEvents\",\"logs:DescribeLogStreams\",\"cloudwatch:PutMetricData\"],\"Resource\":\"*\"}]}"
  sleep 10   # IAM tarda unos segundos en propagar un rol nuevo
}

conectar_lambda() {
  # La Lambda puede consultar e invocar este endpoint, y sabe cómo se llama. Se conservan sus demás
  # variables (API_KEY, …): update-function-configuration reemplaza el mapa completo.
  aws iam put-role-policy --role-name "$ROL_LAMBDA" --policy-name invocar-gpu-sagemaker --policy-document "{
    \"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",
      \"Action\":[\"sagemaker:InvokeEndpoint\",\"sagemaker:DescribeEndpoint\"],
      \"Resource\":\"arn:aws:sagemaker:$REGION:$CUENTA:endpoint/$NOMBRE\"}]}"
  python3 - "$LAMBDA" "$NOMBRE" <<'PY'
import json, subprocess, sys
funcion, endpoint = sys.argv[1], sys.argv[2]
actual = json.loads(subprocess.check_output(["aws", "lambda", "get-function-configuration", "--function-name", funcion,
                                             "--query", "Environment.Variables", "--output", "json"]) or b"{}") or {}
if actual.get("SAGEMAKER_ENDPOINT") != endpoint:
    actual["SAGEMAKER_ENDPOINT"] = endpoint
    subprocess.check_call(["aws", "lambda", "update-function-configuration", "--function-name", funcion,
                           "--environment", json.dumps({"Variables": actual})], stdout=subprocess.DEVNULL)
    subprocess.check_call(["aws", "lambda", "wait", "function-updated", "--function-name", funcion])
print(f"Lambda {funcion}: SAGEMAKER_ENDPOINT={endpoint} (variables conservadas: {', '.join(sorted(actual))})")
PY
}

crear() {
  rol_sagemaker
  aws sagemaker create-model --model-name "$NOMBRE" \
    --execution-role-arn "arn:aws:iam::$CUENTA:role/$ROL" \
    --primary-container "Image=$IMAGEN,Environment={PRECISION=auto}" >/dev/null
  aws sagemaker create-endpoint-config --endpoint-config-name "$NOMBRE" --production-variants \
    "VariantName=principal,ModelName=$NOMBRE,InstanceType=$INSTANCIA,InitialInstanceCount=1,ContainerStartupHealthCheckTimeoutInSeconds=600" >/dev/null
  aws sagemaker create-endpoint --endpoint-name "$NOMBRE" --endpoint-config-name "$NOMBRE" >/dev/null
  conectar_lambda
  echo "Creando el endpoint $NOMBRE en $INSTANCIA (tarda unos 8 minutos)…"
  aws sagemaker wait endpoint-in-service --endpoint-name "$NOMBRE"
  estado
}

estado() {
  aws sagemaker describe-endpoint --endpoint-name "$NOMBRE" \
    --query '{endpoint:EndpointName,estado:EndpointStatus,creado:CreationTime,error:FailureReason}' --output table 2>/dev/null \
    || echo "No hay endpoint $NOMBRE (la GPU está apagada)."
}

borrar() {
  aws sagemaker delete-endpoint --endpoint-name "$NOMBRE" 2>/dev/null && echo "Endpoint borrado" || echo "No había endpoint"
  aws sagemaker delete-endpoint-config --endpoint-config-name "$NOMBRE" 2>/dev/null || true
  aws sagemaker delete-model --model-name "$NOMBRE" 2>/dev/null || true
  if [ "${1:-}" = "--todo" ]; then
    aws ecr delete-repository --repository-name "$REPO" --force >/dev/null 2>&1 || true
    aws iam delete-role-policy --role-name "$ROL" --policy-name pelvis-gpu-minimo 2>/dev/null || true
    aws iam delete-role --role-name "$ROL" 2>/dev/null || true
    aws iam delete-role-policy --role-name "$ROL_LAMBDA" --policy-name invocar-gpu-sagemaker 2>/dev/null || true
    echo "También se borraron la imagen, el repositorio ECR y los permisos."
  fi
}

case "${1:-}" in
  imagen) imagen ;;
  imagen-local) imagen_local ;;
  crear) crear ;;
  estado) estado ;;
  borrar) borrar "${2:-}" ;;
  *) echo "Uso: $0 imagen|imagen-local|crear|estado|borrar [--todo]"; exit 1 ;;
esac
