"""Inferencia de un volumen completo con la API de AWS Lambda (``api/server.py``), sin torch.

Streamlit Community Cloud no tiene torch ni el checkpoint. Para un CT subido, el dashboard preprocesa
el volumen ahí mismo y manda los cortes a ``POST /predict_batch`` en lotes de ``lote`` cortes, con
``paralelo`` pedidos a la vez (Lambda atiende cada pedido en una instancia propia). Cada respuesta
trae la semántica, el mapa primario/secundario, la clasificación y las cajas de esos cortes: lo mismo
que ``predict_volume`` en local, así ``demo_export.exportar_volumen`` no distingue de dónde vino.

La primera llamada después de un rato sin uso es lenta (arranque en frío de cada instancia de Lambda,
hasta ~90 s), por eso ``info_remota`` calienta el servicio antes de mandar los lotes.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _pedir(url: str, key: str | None, ruta: str, cuerpo: bytes | None = None, timeout: float = 150.0) -> dict:
    request = urllib.request.Request(url.rstrip("/") + ruta, data=cuerpo, method="POST" if cuerpo else "GET",
                                     headers={"Content-Type": "application/octet-stream", "X-API-Key": key or ""})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"el servicio respondió {error.code}: {error.read()[:200].decode(errors='replace')}") from error


def info_remota(url: str, key: str | None = None) -> dict:
    """Carga el modelo en el servicio (si estaba frío) y devuelve su descripción."""
    data = _pedir(url, key, "/health?deep=1", timeout=180.0)
    return {"checkpoint": data["model"], "parameters": data["parameters"], "tta": data["tta"],
            "canonical": data["canonical"], "device": f"{data['device']} (servicio remoto)",
            "input_size": data.get("input_size", 384), "precision": data.get("precision", "fp32"),
            "precision_origen": data.get("precision_origen", "")}


def _lote(url: str, key: str | None, images: np.ndarray, intentos: int = 3) -> dict:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(images, dtype=np.uint8), allow_pickle=False)
    for intento in range(intentos):
        try:
            payload = _pedir(url, key, "/predict_batch", buffer.getvalue())
            break
        except (RuntimeError, OSError):
            if intento == intentos - 1:
                raise
            time.sleep(3 * (intento + 1))
    with np.load(io.BytesIO(base64.b64decode(payload["arrays_npz_b64"]))) as packed:
        return {"semantic": packed["semantic"], "primary_secondary": packed["primary_secondary"],
                "classification": packed["classification"], "boxes_raw": payload["boxes_raw"],
                "boxes_refined": payload["boxes_refined"]}


def predictor_remoto(url: str, key: str | None, lote: int = 8, paralelo: int = 4) -> tuple[dict, object]:
    """Como ``demo_export.predictor_local``, pero con la API. Devuelve (info, predecir)."""
    info = info_remota(url, key)

    def predecir(images: np.ndarray, progress=None) -> dict:
        started = time.perf_counter()
        inicios = list(range(0, len(images), lote))
        partes: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=paralelo) as pool:
            futuros = {pool.submit(_lote, url, key, images[i:i + lote]): i for i in inicios}
            for futuro in futuros:
                partes[futuros[futuro]] = futuro.result()
                if progress:
                    progress(min(len(partes) * lote, len(images)), len(images))
        orden = [partes[i] for i in inicios]
        return {"semantic": np.concatenate([p["semantic"] for p in orden]),
                "primary_secondary": np.concatenate([p["primary_secondary"] for p in orden]),
                "classification": np.concatenate([p["classification"] for p in orden]),
                "boxes_raw": [b for p in orden for b in p["boxes_raw"]],
                "boxes_refined": [b for p in orden for b in p["boxes_refined"]],
                "seconds": time.perf_counter() - started}
    return info, predecir
