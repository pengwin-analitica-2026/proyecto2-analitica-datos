#!/usr/bin/env bash
# GPU bajo demanda para la demo: la misma API (api/server.py) en un endpoint de SageMaker con una
# NVIDIA T4 (ml.g4dn.xlarge, ~0.74 USD/hora en us-east-2). Cobra por hora MIENTRAS EXISTA el endpoint:
# se enciende para probar o presentar y se apaga (se borra) después.
#
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh imagen    # GitHub Actions construye y sube la imagen (~15 min)
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh preparar  # una vez: modelo, configuración, roles y la Lambda
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh encender  # crea el endpoint (~8 min) y programa su apagado
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh apagar    # borra el endpoint y cancela el apagado programado
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh crear     # preparar + encender + esperar a que esté lista
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh estado
#   AWS_PROFILE=almia bash deploy/aws/gpu_sagemaker.sh borrar    # apaga y borra también la configuración y el modelo
#
# Encender y apagar también se hace desde el dashboard (barra lateral, «GPU (SageMaker)») con la clave de
# administración GPU_ADMIN_KEY, que «preparar» genera y guarda en ~/.config/pengwin/gpu_admin_key. Se
# encienda por donde se encienda, EventBridge Scheduler borra el endpoint a las HORAS_APAGADO horas (4).
#
# La imagen (~3 GB) la construye GitHub Actions (.github/workflows/build-gpu.yml): desde una conexión de casa
# la subida a ECR se corta. «imagen-local» la construye y sube desde este computador.
#
# La Lambda de la API hace de puente (cabecera X-Destino: gpu). La imagen, el repositorio ECR, el modelo,
# la configuración y los roles se conservan (no cobran por hora); «borrar --todo» también los quita.
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
ROL_APAGADO=pelvis-gpu-apagado
HORAS_APAGADO="${HORAS_APAGADO:-4}"
CLAVE_ADMIN="$HOME/.config/pengwin/gpu_admin_key"
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
}

rol_apagado() {
  # EventBridge Scheduler asume este rol para borrar el endpoint a la hora programada. Solo puede eso.
  if ! aws iam get-role --role-name "$ROL_APAGADO" >/dev/null 2>&1; then
    aws iam create-role --role-name "$ROL_APAGADO" --assume-role-policy-document "{
      \"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"scheduler.amazonaws.com\"},
      \"Action\":\"sts:AssumeRole\",\"Condition\":{\"StringEquals\":{\"aws:SourceAccount\":\"$CUENTA\"}}}]}" >/dev/null
  fi
  aws iam put-role-policy --role-name "$ROL_APAGADO" --policy-name borrar-endpoint-gpu --policy-document "{
    \"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"sagemaker:DeleteEndpoint\",
    \"Resource\":\"arn:aws:sagemaker:$REGION:$CUENTA:endpoint/$NOMBRE\"}]}"
}

conectar_lambda() {
  # La Lambda puede consultar, invocar, crear (desde la configuración ya hecha) y borrar ESTE endpoint,
  # programar su apagado y pasarle al programador el rol de apagado; nada más.
  aws iam put-role-policy --role-name "$ROL_LAMBDA" --policy-name invocar-gpu-sagemaker --policy-document "{
    \"Version\":\"2012-10-17\",\"Statement\":[
      {\"Effect\":\"Allow\",\"Action\":[\"sagemaker:InvokeEndpoint\",\"sagemaker:DescribeEndpoint\",\"sagemaker:CreateEndpoint\",\"sagemaker:DeleteEndpoint\"],
       \"Resource\":[\"arn:aws:sagemaker:$REGION:$CUENTA:endpoint/$NOMBRE\",\"arn:aws:sagemaker:$REGION:$CUENTA:endpoint-config/$NOMBRE\"]},
      {\"Effect\":\"Allow\",\"Action\":[\"scheduler:CreateSchedule\",\"scheduler:UpdateSchedule\",\"scheduler:DeleteSchedule\",\"scheduler:GetSchedule\"],
       \"Resource\":\"arn:aws:scheduler:$REGION:$CUENTA:schedule/default/apagar-$NOMBRE\"},
      {\"Effect\":\"Allow\",\"Action\":\"iam:PassRole\",\"Resource\":\"arn:aws:iam::$CUENTA:role/$ROL_APAGADO\",
       \"Condition\":{\"StringEquals\":{\"iam:PassedToService\":\"scheduler.amazonaws.com\"}}}]}"
  # Variables de la Lambda: se agregan las de la GPU y se conservan las demás (API_KEY, …), porque
  # update-function-configuration reemplaza el mapa completo. El deploy de main solo cambia el código,
  # así que estas variables sobreviven a cada push.
  mkdir -p "$(dirname "$CLAVE_ADMIN")"
  python3 deploy/aws/lambda_variables_gpu.py "$LAMBDA" "$NOMBRE" "arn:aws:iam::$CUENTA:role/$ROL_APAGADO" \
    "$HORAS_APAGADO" "$CLAVE_ADMIN"
}

preparar() {
  # Lo que no cobra por hora y se crea una sola vez. Encender después es solo crear el endpoint.
  rol_sagemaker
  rol_apagado
  # IAM tarda en propagar un rol o una política nuevos: CreateModel comprueba que el rol pueda bajar la
  # imagen y falla con «cannot pull» si se llama enseguida. Se reintenta unas veces.
  local intento
  for intento in 1 2 3 4 5 6; do
    aws sagemaker describe-model --model-name "$NOMBRE" >/dev/null 2>&1 && break
    aws sagemaker create-model --model-name "$NOMBRE" \
         --execution-role-arn "arn:aws:iam::$CUENTA:role/$ROL" \
         --primary-container "Image=$IMAGEN,Environment={PRECISION=auto}" >/dev/null 2>&1 && break
    [ "$intento" = 6 ] && { echo "No se pudo crear el modelo: el rol $ROL no puede bajar $IMAGEN"; exit 1; }
    echo "Esperando a que IAM propague los permisos del rol…"; sleep 15
  done
  aws sagemaker describe-endpoint-config --endpoint-config-name "$NOMBRE" >/dev/null 2>&1 \
    || aws sagemaker create-endpoint-config --endpoint-config-name "$NOMBRE" --production-variants \
         "VariantName=principal,ModelName=$NOMBRE,InstanceType=$INSTANCIA,InitialInstanceCount=1,ContainerStartupHealthCheckTimeoutInSeconds=600" >/dev/null
  conectar_lambda
}

con_python() {
  # src/gpu_bajo_demanda.py: lo mismo que hace la Lambda cuando se aprieta el botón del dashboard.
  SAGEMAKER_ENDPOINT="$NOMBRE" GPU_SCHEDULER_ROLE_ARN="arn:aws:iam::$CUENTA:role/$ROL_APAGADO" \
    GPU_HORAS_APAGADO="$HORAS_APAGADO" python3 src/gpu_bajo_demanda.py "$1"
}

crear() {
  preparar
  con_python encender
  echo "Creando el endpoint $NOMBRE en $INSTANCIA (tarda unos 8 minutos)…"
  aws sagemaker wait endpoint-in-service --endpoint-name "$NOMBRE"
  estado
}

estado() {
  aws sagemaker describe-endpoint --endpoint-name "$NOMBRE" \
    --query '{endpoint:EndpointName,estado:EndpointStatus,creado:CreationTime,error:FailureReason}' --output table 2>/dev/null \
    || echo "No hay endpoint $NOMBRE (la GPU está apagada)."
  local programado
  programado="$(aws scheduler get-schedule --name "apagar-$NOMBRE" --query ScheduleExpression --output text 2>/dev/null || true)"
  [ -n "$programado" ] && echo "Apagado automático: $programado UTC" || true
}

borrar() {
  con_python apagar >/dev/null 2>&1 || true
  echo "Endpoint borrado (o no había)."
  aws sagemaker delete-endpoint-config --endpoint-config-name "$NOMBRE" 2>/dev/null || true
  aws sagemaker delete-model --model-name "$NOMBRE" 2>/dev/null || true
  if [ "${1:-}" = "--todo" ]; then
    aws ecr delete-repository --repository-name "$REPO" --force >/dev/null 2>&1 || true
    aws iam delete-role-policy --role-name "$ROL" --policy-name pelvis-gpu-minimo 2>/dev/null || true
    aws iam delete-role --role-name "$ROL" 2>/dev/null || true
    aws iam delete-role-policy --role-name "$ROL_APAGADO" --policy-name borrar-endpoint-gpu 2>/dev/null || true
    aws iam delete-role --role-name "$ROL_APAGADO" 2>/dev/null || true
    aws iam delete-role-policy --role-name "$ROL_LAMBDA" --policy-name invocar-gpu-sagemaker 2>/dev/null || true
    echo "También se borraron la imagen, el repositorio ECR y los permisos."
  fi
}

case "${1:-}" in
  imagen) imagen ;;
  imagen-local) imagen_local ;;
  preparar) preparar ;;
  encender) con_python encender ;;
  apagar) con_python apagar ;;
  crear) crear ;;
  estado) estado ;;
  borrar) borrar "${2:-}" ;;
  *) echo "Uso: $0 imagen|imagen-local|preparar|encender|apagar|crear|estado|borrar [--todo]"; exit 1 ;;
esac
