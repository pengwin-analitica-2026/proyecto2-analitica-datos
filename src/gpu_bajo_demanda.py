"""Encender, apagar y consultar la GPU (endpoint de SageMaker) bajo demanda, con apagado automático.

El endpoint (``ml.g4dn.xlarge``, NVIDIA T4) cobra ~0.74 USD por hora **mientras exista**. Encenderlo crea
el endpoint a partir de una configuración que ya existe (``deploy/aws/gpu_sagemaker.sh preparar``) y
programa en EventBridge Scheduler una tarea única que lo **borra a las ``HORAS_APAGADO`` horas** (4 por
defecto): si nadie lo apaga, AWS lo apaga solo. Apagarlo borra el endpoint y cancela la tarea. El modelo,
la configuración, la imagen y los roles se conservan (no cobran por hora).

Lo usan la API (botón del dashboard, ``POST /gpu/encender`` y ``/gpu/apagar``) y el script de despliegue.
Configuración por variables de entorno: ``SAGEMAKER_ENDPOINT``, ``GPU_ENDPOINT_CONFIG`` (por defecto el
mismo nombre), ``GPU_SCHEDULER_ROLE_ARN`` (rol que usa el programador para borrar el endpoint) y
``GPU_HORAS_APAGADO``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

HORAS_APAGADO = float(os.environ.get("GPU_HORAS_APAGADO", "4"))


def _nombre() -> str:
    nombre = os.environ.get("SAGEMAKER_ENDPOINT", "")
    if not nombre:
        raise RuntimeError("La GPU no está configurada (falta SAGEMAKER_ENDPOINT).")
    return nombre


def _tarea(endpoint: str) -> str:
    return f"apagar-{endpoint}"


def _clientes():
    import boto3
    return boto3.client("sagemaker"), boto3.client("scheduler")


def estado() -> dict:
    """{estado: InService|Creating|Deleting|Failed|None, apagado_programado: ISO o None, horas_apagado}."""
    sagemaker, scheduler = _clientes()
    endpoint = _nombre()
    try:
        valor = sagemaker.describe_endpoint(EndpointName=endpoint)["EndpointStatus"]
    except Exception:
        valor = None
    apagado = None
    try:
        expresion = scheduler.get_schedule(Name=_tarea(endpoint))["ScheduleExpression"]   # at(2026-10-17T22:00:00)
        apagado = expresion[3:-1] + "Z"
    except Exception:
        pass
    return {"endpoint": endpoint, "estado": valor, "apagado_programado": apagado, "horas_apagado": HORAS_APAGADO}


def encender(horas: float | None = None, ahora: datetime | None = None) -> dict:
    """Crea el endpoint (si no existe) y programa su borrado a las ``horas`` horas. Idempotente."""
    sagemaker, scheduler = _clientes()
    endpoint = _nombre()
    horas = HORAS_APAGADO if horas is None else horas
    try:
        actual = sagemaker.describe_endpoint(EndpointName=endpoint)["EndpointStatus"]
    except Exception:
        actual = None
    if actual in (None, "Failed"):
        if actual == "Failed":
            sagemaker.delete_endpoint(EndpointName=endpoint)
            sagemaker.get_waiter("endpoint_deleted").wait(EndpointName=endpoint)
        sagemaker.create_endpoint(EndpointName=endpoint,
                                  EndpointConfigName=os.environ.get("GPU_ENDPOINT_CONFIG", endpoint))
    elif actual == "Deleting":
        raise RuntimeError("La GPU se está apagando; espera un minuto y vuelve a intentarlo.")
    fin = (ahora or datetime.now(timezone.utc)) + timedelta(hours=horas)
    tarea = {"Name": _tarea(endpoint), "ScheduleExpression": f"at({fin:%Y-%m-%dT%H:%M:%S})",
             "ScheduleExpressionTimezone": "UTC", "FlexibleTimeWindow": {"Mode": "OFF"},
             "ActionAfterCompletion": "DELETE",
             "Description": f"Apagado automático de la GPU de la demo a las {horas:g} h",
             "Target": {"Arn": "arn:aws:scheduler:::aws-sdk:sagemaker:deleteEndpoint",
                        "RoleArn": os.environ["GPU_SCHEDULER_ROLE_ARN"],
                        "Input": json.dumps({"EndpointName": endpoint})}}
    try:
        scheduler.create_schedule(**tarea)
    except scheduler.exceptions.ConflictException:
        scheduler.update_schedule(**tarea)           # ya estaba encendida: se corre el apagado
    return {"endpoint": endpoint, "estado": "Creating" if actual in (None, "Failed") else actual,
            "apagado_programado": f"{fin:%Y-%m-%dT%H:%M:%S}Z", "horas_apagado": horas}


def apagar() -> dict:
    """Borra el endpoint (deja de cobrar) y cancela el apagado programado."""
    sagemaker, scheduler = _clientes()
    endpoint = _nombre()
    borrado = True
    try:
        sagemaker.delete_endpoint(EndpointName=endpoint)
    except Exception:
        borrado = False                              # ya estaba apagada
    try:
        scheduler.delete_schedule(Name=_tarea(endpoint))
    except Exception:
        pass
    return {"endpoint": endpoint, "estado": "Deleting" if borrado else None, "apagado_programado": None}


if __name__ == "__main__":                           # uso desde el script: python src/gpu_bajo_demanda.py encender
    import sys
    accion = sys.argv[1] if len(sys.argv) > 1 else "estado"
    print(json.dumps({"estado": estado, "encender": encender, "apagar": apagar}[accion](), indent=1))
