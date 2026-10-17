"""API de inferencia por corte para el dashboard (se despliega en AWS).

Un solo modelo cargado al arrancar. El dashboard (Streamlit Cloud) manda un corte uint8 [S,S] ya
preprocesado y recibe la semántica, las cajas y la latencia del modelo.

Orientación: H8 espera cortes en orientación canónica (``src/orientacion.py``). El servidor no gira
nada: los cortes que manda el dashboard salen de ``demo_export``, que ya enderezó el volumen
(giro de 180° según la cabecera y, si el paciente estaba girado en el escáner, el giro axial con
``normalizacion.normalizar``). Un corte suelto no basta para estimar ese giro, que necesita el
volumen completo; por eso se corrige al exportar el caso y no aquí.
``/health?deep=1`` informa ``canonical`` para que el cliente lo compruebe.

Precisión (``PRECISION``, por defecto ``auto``): ``auto`` usa int8 en CPU (la Lambda) y fp16 en GPU;
también ``fp32``, ``int8`` o ``fp16``. Es el mismo H8 con el mismo Dice; solo cambia la velocidad (ver
``src/precision.py`` y ``docs/cuantizacion.md``). ``/health?deep=1`` informa cuál se está usando.

    MODEL_PATH=outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth API_KEY=<clave> uvicorn api.server:app --port 8080
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "outputs" / "s10" / "h8_aumm_noise01_noema_w32_e60_s42.pth"))
API_KEY = os.environ.get("API_KEY", "")
TTA = os.environ.get("TTA", "0") == "1"
PRECISION = os.environ.get("PRECISION", "auto")
MAX_BYTES = 2_000_000          # un corte 384x384 uint8 pesa ~150 kB
MAX_LOTE = 16                  # /predict_batch: 16 cortes son ~2.4 MB; Lambda acepta hasta 6 MB por pedido

app = FastAPI(title="Inferencia por corte · fracturas pélvicas", docs_url=None, redoc_url=None)
_state: dict = {}


def _load() -> dict:
    if not _state:
        from predict_volume import load_pipeline, predict_slice
        model, device, info = load_pipeline(MODEL_PATH, tta=TTA, precision=PRECISION)
        _state.update(model=model, device=device, info=info)
        warm = np.zeros((info["input_size"], info["input_size"]), np.uint8)
        predict_slice(model, device, warm)                       # calentamiento
    return _state


def _check_key(key: str | None) -> None:
    if not API_KEY or not key or not hmac.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="clave inválida")


def _health_deep() -> dict:
    state = _load()
    return {"ok": True, "model": state["info"]["checkpoint"], "device": str(state["device"]),
            "parameters": state["info"]["parameters"], "tta": state["info"]["tta"],
            "canonical": state["info"]["canonical"], "input_size": state["info"]["input_size"],
            "precision": state["info"]["precision"], "precision_origen": state["info"]["precision_origen"]}


def _slice(body: bytes) -> dict:
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="corte demasiado grande")
    try:
        image = np.load(io.BytesIO(body), allow_pickle=False)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"no pude leer el corte (.npy): {error}") from error
    state = _load()
    size = state["info"]["input_size"]
    if image.dtype != np.uint8 or image.shape != (size, size):
        raise HTTPException(status_code=422, detail=f"se espera uint8 de {size}x{size}")
    from predict_volume import predict_slice
    result = predict_slice(state["model"], state["device"], image)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, semantic=result["semantic"][0])
    return {"semantic_npz_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "boxes_refined": result["boxes_refined"][0], "boxes_raw": result["boxes_raw"][0],
            "latency_ms": result["latency_ms"], "device": str(state["device"]),
            "model": state["info"]["checkpoint"], "tta": state["info"]["tta"],
            "precision": state["info"]["precision"]}


def _batch(body: bytes) -> dict:
    if len(body) > MAX_LOTE * MAX_BYTES // 10:
        raise HTTPException(status_code=413, detail=f"lote demasiado grande (máximo {MAX_LOTE} cortes)")
    try:
        images = np.load(io.BytesIO(body), allow_pickle=False)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"no pude leer los cortes (.npy): {error}") from error
    state = _load()
    size = state["info"]["input_size"]
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (size, size) \
            or not 1 <= len(images) <= MAX_LOTE:
        raise HTTPException(status_code=422, detail=f"se espera uint8 de [1..{MAX_LOTE}, {size}, {size}]")
    from predict_volume import predict_volume
    result = predict_volume(state["model"], images, state["device"], batch_size=len(images))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, semantic=result["semantic"], primary_secondary=result["primary_secondary"],
                        classification=result["classification"])
    return {"arrays_npz_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "boxes_refined": result["boxes_refined"], "boxes_raw": result["boxes_raw"],
            "seconds": result["seconds"], "device": str(state["device"]),
            "model": state["info"]["checkpoint"], "tta": state["info"]["tta"],
            "precision": state["info"]["precision"]}


# --------------------------------------------------------------------------- GPU en SageMaker
# Un endpoint de SageMaker no tiene URL pública: se invoca con credenciales de AWS. La Lambda (que
# tiene rol) hace de puente: si el pedido trae la cabecera «X-Destino: gpu» y está configurado
# SAGEMAKER_ENDPOINT, lo reenvía al endpoint (la misma API en una imagen con CUDA, api/Dockerfile.gpu)
# y devuelve su respuesta. Así el dashboard no necesita credenciales nuevas. El endpoint se crea y se
# borra con deploy/aws/gpu_sagemaker.sh, porque cobra por hora mientras exista.
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "")
_GPU_ESTADO: dict = {"valor": None, "hasta": 0.0}


def _gpu_estado() -> str | None:
    """Estado del endpoint de GPU («InService», «Creating», …) o None si no existe; se guarda 60 s."""
    if not SAGEMAKER_ENDPOINT:
        return None
    if time.time() < _GPU_ESTADO["hasta"]:
        return _GPU_ESTADO["valor"]
    try:
        import boto3
        valor = boto3.client("sagemaker").describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT)["EndpointStatus"]
    except Exception:
        valor = None
    _GPU_ESTADO.update(valor=valor, hasta=time.time() + 60)
    return valor


def _gpu(ruta: str, body: bytes = b"") -> dict:
    if _gpu_estado() != "InService":
        raise HTTPException(status_code=503, detail="la GPU (SageMaker) no está encendida")
    import boto3
    inicio = time.perf_counter()
    try:
        respuesta = boto3.client("sagemaker-runtime").invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT, ContentType="application/octet-stream",
            CustomAttributes=f"ruta={ruta}", Body=body or b"{}")
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"la GPU respondió con error: {str(error)[:200]}") from error
    datos = json.loads(respuesta["Body"].read())
    datos["via"] = f"SageMaker {SAGEMAKER_ENDPOINT}"
    datos["puente_ms"] = (time.perf_counter() - inicio) * 1000.0
    return datos


def _es_gpu(destino: str | None) -> bool:
    return (destino or "").lower() == "gpu"


@app.get("/health")
def health(deep: int = 0, x_destino: str | None = Header(default=None),
           x_api_key: str | None = Header(default=None)) -> dict:
    # Sin ?deep=1 responde al instante: el adaptador de Lambda lo usa como chequeo de arranque (10 s)
    # y cargar torch + el modelo tarda más en frío. Con ?deep=1 carga el modelo y lo reporta.
    if not deep:
        return {"ok": True, "loaded": bool(_state), "gpu": _gpu_estado()}
    if _es_gpu(x_destino):
        _check_key(x_api_key)
        return _gpu("health")
    return _health_deep()


@app.post("/predict_slice")
async def predict_slice_endpoint(request: Request, x_api_key: str | None = Header(default=None),
                                 x_destino: str | None = Header(default=None)) -> dict:
    _check_key(x_api_key)
    body = await request.body()
    return _gpu("predict_slice", body) if _es_gpu(x_destino) else _slice(body)


@app.post("/predict_batch")
async def predict_batch_endpoint(request: Request, x_api_key: str | None = Header(default=None),
                                 x_destino: str | None = Header(default=None)) -> dict:
    """Varios cortes uint8 [n,S,S] (.npy) → semántica, primario/secundario, clasificación y cajas.

    Lo usa el dashboard para inferir un CT subido completo (``src/inferencia_remota.py``). Los cortes
    llegan ya preprocesados y en la orientación canónica; el servidor no gira nada.
    """
    _check_key(x_api_key)
    body = await request.body()
    return _gpu("predict_batch", body) if _es_gpu(x_destino) else _batch(body)


# Rutas que exige SageMaker para un contenedor propio. No piden X-API-Key: al endpoint solo se llega
# con credenciales de AWS (la Lambda, con su rol). La ruta interna va en CustomAttributes («ruta=…»).
@app.get("/ping")
def ping() -> dict:
    return {"ok": True}


@app.post("/invocations")
async def invocations(request: Request,
                      x_amzn_sagemaker_custom_attributes: str | None = Header(default=None)) -> dict:
    atributos = dict(p.split("=", 1) for p in (x_amzn_sagemaker_custom_attributes or "").split(",") if "=" in p)
    ruta = atributos.get("ruta", "predict_slice")
    body = await request.body()
    if ruta == "health":
        return _health_deep()
    if ruta == "predict_batch":
        return _batch(body)
    if ruta == "predict_slice":
        return _slice(body)
    raise HTTPException(status_code=404, detail=f"ruta desconocida: {ruta}")


# --------------------------------------------------------------------------- casos de PENGWIN en S3
# El dashboard (Streamlit Cloud) no tiene credenciales de AWS: le pide a la API la lista de casos y un
# enlace temporal para descargar cada archivo. La Lambda lee el bucket con su propio rol (solo lectura
# de crudos/ y exportados/), y estos endpoints exigen la misma X-API-Key que la inferencia.
PREFIJOS_S3 = ("crudos/", "exportados/")


def _clave_valida(clave: str) -> str:
    if not clave.startswith(PREFIJOS_S3) or ".." in clave or "//" in clave:
        raise HTTPException(status_code=422, detail="solo se puede leer crudos/ y exportados/")
    return clave


@app.get("/s3/listar")
def s3_listar(prefijo: str, x_api_key: str | None = Header(default=None)) -> dict:
    _check_key(x_api_key)
    import almacen_s3
    return {"nombres": almacen_s3.prefijos_directo(_clave_valida(prefijo))}


@app.get("/s3/url")
def s3_url(clave: str, x_api_key: str | None = Header(default=None)) -> dict:
    """Enlace firmado de 15 minutos para descargar ``clave`` del bucket (404 si no existe)."""
    _check_key(x_api_key)
    clave = _clave_valida(clave)
    import almacen_s3
    cliente = almacen_s3.cliente()
    try:
        cliente.head_object(Bucket=almacen_s3.BUCKET, Key=clave)
    except Exception as error:
        raise HTTPException(status_code=404, detail=f"no existe {clave}") from error
    return {"url": cliente.generate_presigned_url("get_object", Params={"Bucket": almacen_s3.BUCKET, "Key": clave},
                                                  ExpiresIn=900)}
