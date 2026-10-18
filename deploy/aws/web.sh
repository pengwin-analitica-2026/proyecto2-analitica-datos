#!/usr/bin/env bash
# Página web de AWS (web/): S3 + CloudFront para la página y los resultados, y la Lambda pelvis-web (10 GB,
# 15 min) para la API y el procesamiento. No toca la Lambda pelvis-api ni el dashboard de Streamlit.
#
#   AWS_PROFILE=almia bash deploy/aws/web.sh preparar   # una vez: bucket, rol, ECR, Lambda, URL y CloudFront
#   AWS_PROFILE=almia bash deploy/aws/web.sh imagen     # construye y sube la imagen; actualiza la Lambda
#   AWS_PROFILE=almia bash deploy/aws/web.sh pagina     # compila el frontend y lo sube con los casos de ejemplo
#   AWS_PROFILE=almia bash deploy/aws/web.sh todo       # los tres
#   AWS_PROFILE=almia bash deploy/aws/web.sh estado     # direcciones y configuración
#   AWS_PROFILE=almia bash deploy/aws/web.sh borrar     # borra todo lo anterior (al terminar el curso)
#
# Ver docs/web_aws.md.
set -euo pipefail

REGION="${AWS_REGION:-us-east-2}"
CUENTA="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${WEB_BUCKET:-pengwin-web-$CUENTA}"
PENGWIN_BUCKET="${PENGWIN_BUCKET:-pengwin-ct-$CUENTA}"
FUNCION=pelvis-web
ROL=pelvis-web-lambda-exec
REPO=pelvis-web
MODELO=h8_aumm_noise01_noema_w32_e60_s42
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python3}"
ESTADO="$RAIZ/deploy/aws/.web_estado"      # id de la distribución de CloudFront (no se versiona)

aws_() { aws --region "$REGION" "$@"; }

distribucion() { [ -f "$ESTADO" ] && cat "$ESTADO" || true; }
dominio() { aws cloudfront get-distribution --id "$(distribucion)" --query Distribution.DomainName --output text; }
url_api() { aws_ lambda get-function-url-config --function-name "$FUNCION" --query FunctionUrl --output text; }

preparar() {
  echo "== bucket $BUCKET"
  if ! aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    aws_ s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
  aws_ s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  # Los CT subidos y el estado de los trabajos se borran solos; los resultados de subidas, a la semana.
  aws_ s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration '{"Rules": [
    {"ID": "subidas", "Status": "Enabled", "Filter": {"Prefix": "subidas/"}, "Expiration": {"Days": 1}},
    {"ID": "trabajos", "Status": "Enabled", "Filter": {"Prefix": "trabajos/"}, "Expiration": {"Days": 7}},
    {"ID": "resultados-subidas", "Status": "Enabled", "Filter": {"Prefix": "resultados/'"$MODELO"'/subidas/"}, "Expiration": {"Days": 7}}]}'

  echo "== ECR $REPO"
  aws_ ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 || \
    aws_ ecr create-repository --repository-name "$REPO" --image-scanning-configuration scanOnPush=true >/dev/null
  aws_ ecr put-lifecycle-policy --repository-name "$REPO" --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' >/dev/null

  echo "== rol $ROL"
  if ! aws iam get-role --role-name "$ROL" >/dev/null 2>&1; then
    aws iam create-role --role-name "$ROL" --assume-role-policy-document file://"$RAIZ/deploy/aws/lambda-trust.json" >/dev/null
    aws iam attach-role-policy --role-name "$ROL" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  fi
  sed -e "s/{BUCKET}/$BUCKET/g" -e "s/{PENGWIN_BUCKET}/$PENGWIN_BUCKET/g" -e "s/{REGION}/$REGION/g" \
      -e "s/{CUENTA}/$CUENTA/g" -e "s/{FUNCION}/$FUNCION/g" "$RAIZ/deploy/aws/web-lambda-policy.json" > /tmp/pelvis-web-policy.json
  aws iam put-role-policy --role-name "$ROL" --policy-name pelvis-web --policy-document file:///tmp/pelvis-web-policy.json

  if ! aws_ lambda get-function --function-name "$FUNCION" >/dev/null 2>&1; then
    echo "== primera imagen (para poder crear la función)"
    construir_y_subir
    sleep 10                                     # el rol recién creado tarda en propagarse
    INFERENCE_URL="$(aws_ lambda get-function-url-config --function-name pelvis-api --query FunctionUrl --output text 2>/dev/null || true)"
    INFERENCE_KEY="$(aws_ lambda get-function-configuration --function-name pelvis-api --query 'Environment.Variables.API_KEY' --output text 2>/dev/null || true)"
    aws_ lambda create-function --function-name "$FUNCION" --package-type Image \
      --code ImageUri="$(cat /tmp/pelvis-web-imagen)" --role "arn:aws:iam::$CUENTA:role/$ROL" \
      --memory-size 10240 --timeout 900 --ephemeral-storage Size=10240 --architectures x86_64 \
      --environment "Variables={WEB_BUCKET=$BUCKET,WEB_REGION=$REGION,PENGWIN_BUCKET=$PENGWIN_BUCKET,INFERENCE_URL=${INFERENCE_URL%/},INFERENCE_KEY=$INFERENCE_KEY}" >/dev/null
    aws_ lambda wait function-active-v2 --function-name "$FUNCION"
  fi
  # Un trabajo que falla no se reintenta (el error ya queda en trabajos/<id>.json).
  aws_ lambda put-function-event-invoke-config --function-name "$FUNCION" --maximum-retry-attempts 0 >/dev/null

  echo "== CloudFront"
  if [ -z "$(distribucion)" ]; then
    OAC=$(aws cloudfront create-origin-access-control --origin-access-control-config \
      "Name=$BUCKET,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" \
      --query OriginAccessControl.Id --output text)
    cat > /tmp/pelvis-web-cf.json <<JSON
{
  "CallerReference": "pelvis-web-$(date +%s)",
  "Comment": "PENGWIN web (pagina y resultados)",
  "Enabled": true,
  "DefaultRootObject": "index.html",
  "PriceClass": "PriceClass_100",
  "HttpVersion": "http2and3",
  "Origins": {"Quantity": 1, "Items": [{
    "Id": "s3", "DomainName": "$BUCKET.s3.$REGION.amazonaws.com", "OriginAccessControlId": "$OAC",
    "S3OriginConfig": {"OriginAccessIdentity": ""}}]},
  "DefaultCacheBehavior": {
    "TargetOriginId": "s3", "ViewerProtocolPolicy": "redirect-to-https", "Compress": true,
    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
    "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
  "CustomErrorResponses": {"Quantity": 1, "Items": [{"ErrorCode": 403, "ErrorCachingMinTTL": 5}]}
}
JSON
    aws cloudfront create-distribution --distribution-config file:///tmp/pelvis-web-cf.json \
      --query Distribution.Id --output text > "$ESTADO"
  fi
  ARN="arn:aws:cloudfront::$CUENTA:distribution/$(distribucion)"
  # CloudFront solo puede leer la página, los resultados y datos/ (no los CT subidos ni los trabajos).
  aws_ s3api put-bucket-policy --bucket "$BUCKET" --policy '{"Version": "2012-10-17", "Statement": [{
    "Effect": "Allow", "Principal": {"Service": "cloudfront.amazonaws.com"}, "Action": "s3:GetObject",
    "Resource": ["arn:aws:s3:::'"$BUCKET"'/index.html", "arn:aws:s3:::'"$BUCKET"'/favicon.svg",
                 "arn:aws:s3:::'"$BUCKET"'/assets/*", "arn:aws:s3:::'"$BUCKET"'/resultados/*",
                 "arn:aws:s3:::'"$BUCKET"'/datos/*"],
    "Condition": {"StringEquals": {"AWS:SourceArn": "'"$ARN"'"}}}]}'
  ORIGEN="https://$(dominio)"

  echo "== URL de la función (CORS solo para $ORIGEN y localhost)"
  CORS='{"AllowOrigins": ["'"$ORIGEN"'", "http://localhost:5173", "http://localhost:8080"], "AllowMethods": ["*"],
         "AllowHeaders": ["content-type", "x-destino", "x-admin-key"], "MaxAge": 3600}'
  if aws_ lambda get-function-url-config --function-name "$FUNCION" >/dev/null 2>&1; then
    aws_ lambda update-function-url-config --function-name "$FUNCION" --cors "$CORS" >/dev/null
  else
    aws_ lambda create-function-url-config --function-name "$FUNCION" --auth-type NONE --cors "$CORS" >/dev/null
    aws_ lambda add-permission --function-name "$FUNCION" --statement-id url-publica \
      --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
    # La URL pública necesita también InvokeFunction (ver docs/despliegue_aws_streamlit.md); el CLI viejo
    # no tiene la opción, así que va con boto3.
    "$PY" - "$FUNCION" "$REGION" <<'PYTHON'
import sys, boto3
boto3.client("lambda", region_name=sys.argv[2]).add_permission(
    FunctionName=sys.argv[1], StatementId="url-publica-invoke", Action="lambda:InvokeFunction",
    Principal="*", InvokedViaFunctionUrl=True)
PYTHON
  fi
  # El bucket acepta las subidas firmadas directo desde la página.
  aws_ s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration '{"CORSRules": [{
    "AllowedOrigins": ["'"$ORIGEN"'", "http://localhost:5173", "http://localhost:8080"],
    "AllowedMethods": ["POST"], "AllowedHeaders": ["*"], "MaxAgeSeconds": 3600}]}'
  estado
}

construir_y_subir() {
  local registro="$CUENTA.dkr.ecr.$REGION.amazonaws.com"
  local imagen="$registro/$REPO:$(git -C "$RAIZ" rev-parse --short HEAD)-$(date +%Y%m%d%H%M)"
  aws_ ecr get-login-password | docker login --username AWS --password-stdin "$registro" >/dev/null
  docker build --platform linux/amd64 --provenance=false -f "$RAIZ/web/Dockerfile" -t "$imagen" "$RAIZ"
  # Desde una conexión de casa la subida de la capa de torch a veces se corta: se reintenta.
  for intento in 1 2 3 4 5; do docker push "$imagen" && break || { echo "reintento $intento"; sleep 5; }; done
  echo "$imagen" > /tmp/pelvis-web-imagen
}

imagen() {
  construir_y_subir
  aws_ lambda update-function-code --function-name "$FUNCION" --image-uri "$(cat /tmp/pelvis-web-imagen)" >/dev/null
  aws_ lambda wait function-updated-v2 --function-name "$FUNCION"
  curl -fsS "$(url_api)health" && echo
}

pagina() {
  local tmp; tmp="$(mktemp -d)"
  echo "== casos de ejemplo y resumen de métricas"
  (cd "$RAIZ" && "$PY" src/paquete_web.py outputs/demo/$MODELO/* --destino "$tmp/resultados/$MODELO/ejemplos" \
     --resumen "$tmp/datos/resumen.json")
  aws_ s3 sync "$tmp/resultados" "s3://$BUCKET/resultados" --only-show-errors --cache-control "public, max-age=86400"
  aws_ s3 cp "$tmp/datos/resumen.json" "s3://$BUCKET/datos/resumen.json" --cache-control "public, max-age=300"
  rm -rf "$tmp"
  echo "== frontend"
  (cd "$RAIZ/web/frontend" && npm ci --no-audit --no-fund && VITE_API_URL="$(url_api)" npm run build)
  aws_ s3 sync "$RAIZ/web/frontend/dist/assets" "s3://$BUCKET/assets" --cache-control "public, max-age=31536000, immutable"
  aws_ s3 cp "$RAIZ/web/frontend/dist/index.html" "s3://$BUCKET/index.html" --cache-control "no-cache"
  aws_ s3 cp "$RAIZ/web/frontend/dist/favicon.svg" "s3://$BUCKET/favicon.svg" --cache-control "public, max-age=86400"
  aws cloudfront create-invalidation --distribution-id "$(distribucion)" --paths "/index.html" "/datos/*" >/dev/null
  estado
}

estado() {
  echo "Página:    https://$(dominio)"
  echo "API:       $(url_api)"
  echo "Bucket:    s3://$BUCKET"
  aws_ lambda get-function-configuration --function-name "$FUNCION" \
    --query '{memoria: MemorySize, tiempo: Timeout, imagen: CodeSha256, estado: LastUpdateStatus}' --output table
}

borrar() {
  read -r -p "Esto borra la página web de AWS (bucket $BUCKET, CloudFront, Lambda $FUNCION, ECR $REPO). ¿Seguro? [s/N] " ok
  [ "$ok" = "s" ] || exit 1
  if [ -n "$(distribucion)" ]; then
    ID="$(distribucion)"; ETAG=$(aws cloudfront get-distribution-config --id "$ID" --query ETag --output text)
    aws cloudfront get-distribution-config --id "$ID" --query DistributionConfig > /tmp/cf.json
    "$PY" -c "import json;d=json.load(open('/tmp/cf.json'));d['Enabled']=False;json.dump(d,open('/tmp/cf.json','w'))"
    aws cloudfront update-distribution --id "$ID" --if-match "$ETAG" --distribution-config file:///tmp/cf.json >/dev/null
    aws cloudfront wait distribution-deployed --id "$ID"
    aws cloudfront delete-distribution --id "$ID" --if-match "$(aws cloudfront get-distribution --id "$ID" --query ETag --output text)"
    rm -f "$ESTADO"
  fi
  aws_ lambda delete-function --function-name "$FUNCION" || true
  aws_ ecr delete-repository --repository-name "$REPO" --force >/dev/null || true
  aws iam delete-role-policy --role-name "$ROL" --policy-name pelvis-web || true
  aws iam detach-role-policy --role-name "$ROL" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole || true
  aws iam delete-role --role-name "$ROL" || true
  aws_ s3 rb "s3://$BUCKET" --force || true
}

case "${1:-}" in
  preparar) preparar ;;
  imagen) imagen ;;
  pagina) pagina ;;
  todo) preparar; imagen; pagina ;;
  estado) estado ;;
  borrar) borrar ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
