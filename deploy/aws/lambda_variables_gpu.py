"""Agrega a la Lambda de la API las variables de la GPU bajo demanda sin borrar las que ya tiene.

    python3 deploy/aws/lambda_variables_gpu.py <función> <endpoint> <rol de apagado> <horas> <archivo de clave>

``update-function-configuration`` reemplaza el mapa de variables completo, así que se lee el actual
(con API_KEY) y se le suman SAGEMAKER_ENDPOINT, GPU_SCHEDULER_ROLE_ARN, GPU_HORAS_APAGADO y
GPU_ADMIN_KEY. La clave de administración se genera una vez y queda en ``<archivo de clave>`` (permisos
600) para quien vaya a encender la GPU desde el dashboard; si la Lambda ya tenía una, se conserva esa.
Nunca se imprime.
"""
import json
import os
import secrets
import subprocess
import sys

funcion, endpoint, rol, horas, archivo = sys.argv[1:]
actual = json.loads(subprocess.check_output(
    ["aws", "lambda", "get-function-configuration", "--function-name", funcion,
     "--query", "Environment.Variables", "--output", "json"]) or b"{}") or {}
guardada = open(archivo).read().strip() if os.path.exists(archivo) else ""
clave = actual.get("GPU_ADMIN_KEY") or guardada or secrets.token_urlsafe(24)
if clave != guardada:
    descriptor = os.open(archivo, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as salida:
        salida.write(clave + "\n")
nuevas = {"SAGEMAKER_ENDPOINT": endpoint, "GPU_SCHEDULER_ROLE_ARN": rol, "GPU_HORAS_APAGADO": horas,
          "GPU_ADMIN_KEY": clave}
if any(actual.get(nombre) != valor for nombre, valor in nuevas.items()):
    actual.update(nuevas)
    subprocess.check_call(["aws", "lambda", "update-function-configuration", "--function-name", funcion,
                           "--environment", json.dumps({"Variables": actual})], stdout=subprocess.DEVNULL)
    subprocess.check_call(["aws", "lambda", "wait", "function-updated", "--function-name", funcion])
print(f"Lambda {funcion}: GPU {endpoint}, apagado automático a las {horas} h. "
      f"Clave de administración en {archivo} (variables: {', '.join(sorted(actual))})")
