"""API de inferencia por corte para el dashboard (se despliega en AWS).

Un solo modelo cargado al arrancar. El dashboard (Streamlit Cloud) manda un corte uint8 [S,S] ya
preprocesado y recibe la semántica, las cajas y la latencia del modelo.

Orientación: H8 espera cortes en orientación canónica (``src/orientacion.py``). El servidor no gira
nada: los cortes que manda el dashboard salen de ``demo_export``, que ya enderezó el volumen
(giro de 180° según la cabecera y, si el paciente estaba girado en el escáner, el giro axial con
``normalizacion.normalizar``). Un corte suelto no basta para estimar ese giro, que necesita el
volumen completo; por eso se corrige al exportar el caso y no aquí.
``/health?deep=1`` informa ``canonical`` para que el cliente lo compruebe.

    MODEL_PATH=outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth API_KEY=<clave> uvicorn api.server:app --port 8080
"""
from __future__ import annotations

import base64
import hmac
import io
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "outputs" / "s10" / "h8_aumm_noise01_noema_w32_e60_s42.pth"))
API_KEY = os.environ.get("API_KEY", "")
TTA = os.environ.get("TTA", "0") == "1"
MAX_BYTES = 2_000_000          # un corte 384x384 uint8 pesa ~150 kB
MAX_LOTE = 16                  # /predict_batch: 16 cortes son ~2.4 MB; Lambda acepta hasta 6 MB por pedido

app = FastAPI(title="Inferencia por corte · fracturas pélvicas", docs_url=None, redoc_url=None)
_state: dict = {}


def _load() -> dict:
    if not _state:
        from predict_volume import load_pipeline, predict_slice
        model, device, info = load_pipeline(MODEL_PATH, tta=TTA)
        _state.update(model=model, device=device, info=info)
        warm = np.zeros((info["input_size"], info["input_size"]), np.uint8)
        predict_slice(model, device, warm)                       # calentamiento
    return _state


def _check_key(key: str | None) -> None:
    if not API_KEY or not key or not hmac.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="clave inválida")


@app.get("/health")
def health(deep: int = 0) -> dict:
    # Sin ?deep=1 responde al instante: el adaptador de Lambda lo usa como chequeo de arranque (10 s)
    # y cargar torch + el modelo tarda más en frío. Con ?deep=1 carga el modelo y lo reporta.
    if not deep:
        return {"ok": True, "loaded": bool(_state)}
    state = _load()
    return {"ok": True, "model": state["info"]["checkpoint"], "device": str(state["device"]),
            "parameters": state["info"]["parameters"], "tta": state["info"]["tta"],
            "canonical": state["info"]["canonical"], "input_size": state["info"]["input_size"]}


@app.post("/predict_slice")
async def predict_slice_endpoint(request: Request, x_api_key: str | None = Header(default=None)) -> dict:
    _check_key(x_api_key)
    body = await request.body()
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
            "model": state["info"]["checkpoint"], "tta": state["info"]["tta"]}


@app.post("/predict_batch")
async def predict_batch_endpoint(request: Request, x_api_key: str | None = Header(default=None)) -> dict:
    """Varios cortes uint8 [n,S,S] (.npy) → semántica, primario/secundario, clasificación y cajas.

    Lo usa el dashboard para inferir un CT subido completo (``src/inferencia_remota.py``). Los cortes
    llegan ya preprocesados y en la orientación canónica; el servidor no gira nada.
    """
    _check_key(x_api_key)
    body = await request.body()
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
            "model": state["info"]["checkpoint"], "tta": state["info"]["tta"]}


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
