"""GPU bajo demanda: encender programa el apagado a las 4 h, apagar lo cancela, y la API pide la clave
de administración. SageMaker y EventBridge Scheduler están simulados: no se toca AWS."""
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gpu_bajo_demanda  # noqa: E402


class Conflicto(Exception):
    pass


class SageMakerFalso:
    def __init__(self, estado=None):
        self.estado, self.llamadas = estado, []

    def describe_endpoint(self, EndpointName):
        if self.estado is None:
            raise RuntimeError("Could not find endpoint")
        return {"EndpointStatus": self.estado}

    def create_endpoint(self, EndpointName, EndpointConfigName):
        self.llamadas.append(("crear", EndpointName, EndpointConfigName))
        self.estado = "Creating"

    def delete_endpoint(self, EndpointName):
        if self.estado is None:
            raise RuntimeError("Could not find endpoint")
        self.llamadas.append(("borrar", EndpointName))
        self.estado = None

    def get_waiter(self, nombre):
        return mock.Mock()


class SchedulerFalso:
    exceptions = mock.Mock(ConflictException=Conflicto)

    def __init__(self):
        self.tareas = {}

    def create_schedule(self, **tarea):
        if tarea["Name"] in self.tareas:
            raise Conflicto()
        self.tareas[tarea["Name"]] = tarea

    def update_schedule(self, **tarea):
        self.tareas[tarea["Name"]] = tarea

    def get_schedule(self, Name):
        return self.tareas[Name]

    def delete_schedule(self, Name):
        del self.tareas[Name]


ENTORNO = {"SAGEMAKER_ENDPOINT": "pelvis-h8-gpu",
           "GPU_SCHEDULER_ROLE_ARN": "arn:aws:iam::123456789012:role/pelvis-gpu-apagado"}
AHORA = datetime(2026, 10, 17, 18, 30, tzinfo=timezone.utc)


class GpuBajoDemanda(unittest.TestCase):
    def setUp(self):
        self.sagemaker, self.scheduler = SageMakerFalso(), SchedulerFalso()
        parches = [mock.patch.dict(os.environ, ENTORNO),
                   mock.patch.object(gpu_bajo_demanda, "_clientes", lambda: (self.sagemaker, self.scheduler))]
        for parche in parches:
            parche.start()
            self.addCleanup(parche.stop)

    def test_encender_crea_el_endpoint_y_programa_el_apagado_a_las_4_horas(self):
        datos = gpu_bajo_demanda.encender(horas=4, ahora=AHORA)
        self.assertEqual(self.sagemaker.llamadas, [("crear", "pelvis-h8-gpu", "pelvis-h8-gpu")])
        tarea = self.scheduler.tareas["apagar-pelvis-h8-gpu"]
        self.assertEqual(tarea["ScheduleExpression"], "at(2026-10-17T22:30:00)")
        self.assertEqual(tarea["ScheduleExpressionTimezone"], "UTC")
        self.assertEqual(tarea["ActionAfterCompletion"], "DELETE")
        self.assertEqual(tarea["Target"]["Arn"], "arn:aws:scheduler:::aws-sdk:sagemaker:deleteEndpoint")
        self.assertEqual(json.loads(tarea["Target"]["Input"]), {"EndpointName": "pelvis-h8-gpu"})
        self.assertEqual(datos["estado"], "Creating")
        self.assertEqual(datos["apagado_programado"], "2026-10-17T22:30:00Z")
        self.assertEqual(gpu_bajo_demanda.estado()["apagado_programado"], "2026-10-17T22:30:00Z")

    def test_encender_otra_vez_no_crea_otro_endpoint_y_corre_el_apagado(self):
        gpu_bajo_demanda.encender(horas=4, ahora=AHORA)
        self.sagemaker.estado = "InService"
        datos = gpu_bajo_demanda.encender(horas=4, ahora=AHORA.replace(hour=20))
        self.assertEqual(len(self.sagemaker.llamadas), 1)
        self.assertEqual(datos["estado"], "InService")
        self.assertEqual(self.scheduler.tareas["apagar-pelvis-h8-gpu"]["ScheduleExpression"], "at(2026-10-18T00:30:00)")

    def test_un_endpoint_fallido_se_recrea_y_uno_que_se_apaga_no(self):
        self.sagemaker.estado = "Failed"
        gpu_bajo_demanda.encender(ahora=AHORA)
        self.assertEqual([c[0] for c in self.sagemaker.llamadas], ["borrar", "crear"])
        self.sagemaker.estado = "Deleting"
        with self.assertRaises(RuntimeError):
            gpu_bajo_demanda.encender(ahora=AHORA)

    def test_apagar_borra_el_endpoint_y_cancela_la_tarea(self):
        gpu_bajo_demanda.encender(ahora=AHORA)
        datos = gpu_bajo_demanda.apagar()
        self.assertEqual(self.sagemaker.llamadas[-1], ("borrar", "pelvis-h8-gpu"))
        self.assertEqual(self.scheduler.tareas, {})
        self.assertEqual(datos["estado"], "Deleting")
        self.assertIsNone(gpu_bajo_demanda.apagar()["estado"])          # ya apagada: no falla
        self.assertEqual(gpu_bajo_demanda.estado()["estado"], None)

    def test_sin_endpoint_configurado(self):
        with mock.patch.dict(os.environ, {"SAGEMAKER_ENDPOINT": ""}):
            with self.assertRaises(RuntimeError):
                gpu_bajo_demanda.encender()


try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi/httpx no instalados")
class ApiGpu(unittest.TestCase):
    """/gpu/*: la X-API-Key no alcanza para encender o apagar; hace falta GPU_ADMIN_KEY."""

    def setUp(self):
        self.entorno_previo = {k: os.environ.get(k) for k in ("API_KEY", "GPU_ADMIN_KEY", "SAGEMAKER_ENDPOINT")}
        self.modulo_previo = sys.modules.pop("api.server", None)
        os.environ.update(API_KEY="clave-api", GPU_ADMIN_KEY="clave-admin", **ENTORNO)
        sys.path.insert(0, str(ROOT))
        import importlib
        server = importlib.import_module("api.server")     # nuevo: lee las claves de este entorno
        self.client = TestClient(server.app)
        self.sagemaker, self.scheduler = SageMakerFalso(), SchedulerFalso()
        parche = mock.patch.object(gpu_bajo_demanda, "_clientes", lambda: (self.sagemaker, self.scheduler))
        parche.start()
        self.addCleanup(parche.stop)

    def tearDown(self):
        for nombre, valor in self.entorno_previo.items():
            if valor is None:
                os.environ.pop(nombre, None)
            else:
                os.environ[nombre] = valor
        # «from api import server» lee el atributo del paquete, no solo sys.modules: se restauran los dos.
        import api
        sys.modules.pop("api.server", None)
        if self.modulo_previo is not None:
            sys.modules["api.server"] = api.server = self.modulo_previo
        elif hasattr(api, "server"):
            del api.server

    def test_claves(self):
        self.assertEqual(self.client.post("/gpu/encender").status_code, 401)
        self.assertEqual(self.client.post("/gpu/encender", headers={"X-API-Key": "clave-api"}).status_code, 403)
        self.assertEqual(self.client.post("/gpu/encender", headers={"X-API-Key": "clave-api",
                                                                     "X-Admin-Key": "otra"}).status_code, 403)
        self.assertEqual(self.sagemaker.llamadas, [])
        self.assertEqual(self.client.get("/gpu/estado").status_code, 401)

    def test_encender_apagar_y_estado(self):
        claves = {"X-API-Key": "clave-api", "X-Admin-Key": "clave-admin"}
        r = self.client.post("/gpu/encender", headers=claves)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["estado"], "Creating")
        self.assertIn("apagar-pelvis-h8-gpu", self.scheduler.tareas)
        estado = self.client.get("/gpu/estado", headers={"X-API-Key": "clave-api"}).json()
        self.assertEqual(estado["estado"], "Creating")
        self.assertTrue(estado["apagado_programado"].endswith("Z"))
        self.assertEqual(self.client.get("/health").json()["gpu"], "Creating")    # caché actualizada
        self.assertEqual(self.client.post("/gpu/apagar", headers=claves).json()["estado"], "Deleting")
        self.assertEqual(self.scheduler.tareas, {})
        self.sagemaker.estado = "Deleting"
        self.assertEqual(self.client.post("/gpu/encender", headers=claves).status_code, 409)


if __name__ == "__main__":
    unittest.main()
