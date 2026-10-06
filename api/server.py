"""API de inferencia por corte para el dashboard (se despliega en AWS).

Un solo modelo cargado al arrancar. El dashboard (Streamlit Cloud) manda un corte uint8 [S,S] ya
preprocesado y recibe la semántica, las cajas y la latencia del modelo.

    MODEL_PATH=outputs/s10/d5_ema_w32_e60_s42.pth API_KEY=<clave> uvicorn api.server:app --port 8080
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

MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "outputs" / "s10" / "d5_ema_w32_e60_s42.pth"))
API_KEY = os.environ.get("API_KEY", "")
TTA = os.environ.get("TTA", "0") == "1"
MAX_BYTES = 2_000_000          # un corte 384x384 uint8 pesa ~150 kB

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
def health() -> dict:
    state = _load()
    return {"ok": True, "model": state["info"]["checkpoint"], "device": str(state["device"]),
            "parameters": state["info"]["parameters"], "tta": state["info"]["tta"]}


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
