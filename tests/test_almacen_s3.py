"""almacen_s3 con un S3 falso en memoria (sin red ni credenciales)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import almacen_s3  # noqa: E402


class S3Falso:
    def __init__(self, objetos: dict[str, bytes]):
        self.objetos = objetos

    def list_objects_v2(self, Bucket, Prefix, Delimiter="/", **_):
        prefijos, contenidos = set(), []
        for clave in self.objetos:
            if not clave.startswith(Prefix):
                continue
            resto = clave[len(Prefix):]
            if Delimiter in resto:
                prefijos.add(Prefix + resto.split(Delimiter)[0] + Delimiter)
            else:
                contenidos.append({"Key": clave})
        return {"CommonPrefixes": [{"Prefix": p} for p in sorted(prefijos)], "Contents": contenidos,
                "IsTruncated": False}

    def download_file(self, bucket, clave, destino):
        if clave not in self.objetos:
            raise FileNotFoundError(clave)
        Path(destino).write_bytes(self.objetos[clave])

    def upload_file(self, origen, bucket, clave):
        self.objetos[clave] = Path(origen).read_bytes()

    def head_object(self, Bucket, Key):
        if Key not in self.objetos:
            raise FileNotFoundError(Key)
        return {}

    def generate_presigned_url(self, operacion, Params, ExpiresIn):
        return f"memoria://{Params['Key']}"


class AlmacenS3(unittest.TestCase):
    def setUp(self):
        self.s3 = S3Falso({"crudos/imagenes/024.mha": b"img", "crudos/etiquetas/024.mha": b"lbl",
                           "crudos/imagenes/044.mha": b"img2",
                           "exportados/h8/044/meta.json": b"{}", "exportados/h8/044/volume.npz": b"v"})
        for parche in (mock.patch.object(almacen_s3, "cliente", lambda: self.s3),
                       mock.patch.object(almacen_s3, "modo", lambda: "directo")):
            parche.start()
            self.addCleanup(parche.stop)

    def test_listar_casos(self):
        casos = {c["caso"]: c for c in almacen_s3.listar_casos("h8")}
        self.assertEqual(set(casos), {"024", "044"})
        self.assertFalse(casos["024"]["precalculado"])
        self.assertTrue(casos["044"]["precalculado"])
        self.assertTrue(casos["044"]["girado"])                 # 044 está en LPI_CASES
        self.assertEqual(casos["024"]["split"], almacen_s3.splits().get("024", "?"))

    def test_descargas(self):
        with tempfile.TemporaryDirectory() as d:
            carpeta = almacen_s3.descargar_exportado("h8", "044", Path(d) / "x")
            self.assertTrue((carpeta / "meta.json").exists())
            self.assertFalse((carpeta / "pasos.npz").exists())   # los opcionales que faltan se saltan
            with self.assertRaises(FileNotFoundError):
                almacen_s3.descargar_exportado("h8", "024", Path(d) / "y")
            imagen, etiqueta = almacen_s3.descargar_crudo("024", Path(d) / "z")
            self.assertEqual(imagen.read_bytes(), b"img")
            self.assertEqual(etiqueta.read_bytes(), b"lbl")
            self.assertIsNone(almacen_s3.descargar_crudo("044", Path(d) / "w")[1])

    def test_subir_exportado(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "meta.json").write_text("{}")
            (Path(d) / "predictions.npz").write_bytes(b"no se sube")
            almacen_s3.subir_exportado(Path(d), "h8", "001")
        self.assertIn("exportados/h8/001/meta.json", self.s3.objetos)
        self.assertNotIn("exportados/h8/001/predictions.npz", self.s3.objetos)



try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi/httpx no instalados")
class AlmacenS3PorLaApi(unittest.TestCase):
    """Sin credenciales de AWS (Streamlit Cloud): el dashboard lee el bucket a través de la API."""

    def setUp(self):
        import io
        import os
        anterior = os.environ.get("API_KEY")
        os.environ["API_KEY"] = "clave-s3"
        sys.path.insert(0, str(ROOT))
        self._olvidar_servidor()
        from api import server
        # Deja el módulo y la variable como estaban: otras pruebas importan la API con su propia clave.
        self.addCleanup(self._olvidar_servidor)
        self.addCleanup(lambda: os.environ.__setitem__("API_KEY", anterior) if anterior is not None
                        else os.environ.pop("API_KEY", None))
        self.api = TestClient(server.app)
        self.s3 = S3Falso({"crudos/imagenes/024.mha": b"img", "exportados/h8/024/meta.json": b"{}",
                           "otros/secreto.txt": b"no"})

        def pedir(url, key, ruta, cuerpo=None, timeout=60.0):
            r = self.api.get(ruta, headers={"X-API-Key": key})
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code}: {r.text}")
            return r.json()

        def abrir(url, timeout=300):
            return io.BytesIO(self.s3.objetos[url.removeprefix("memoria://")])

        import inferencia_remota
        for parche in (mock.patch.object(almacen_s3, "cliente", lambda: self.s3),
                       mock.patch.object(almacen_s3, "hay_credenciales", lambda: False),
                       mock.patch.object(inferencia_remota, "_pedir", pedir),
                       mock.patch.object(almacen_s3.urllib.request, "urlopen", abrir)):
            parche.start()
            self.addCleanup(parche.stop)
        almacen_s3.configurar_api("http://api", "clave-s3")
        self.addCleanup(almacen_s3.configurar_api, None, None)

    @staticmethod
    def _olvidar_servidor():
        sys.modules.pop("api.server", None)
        if "api" in sys.modules and hasattr(sys.modules["api"], "server"):
            delattr(sys.modules["api"], "server")

    def test_listar_y_descargar_sin_credenciales(self):
        self.assertEqual(almacen_s3.modo(), "api")
        casos = {c["caso"]: c for c in almacen_s3.listar_casos("h8")}
        self.assertTrue(casos["024"]["precalculado"])
        with tempfile.TemporaryDirectory() as d:
            imagen, etiqueta = almacen_s3.descargar_crudo("024", Path(d))
            self.assertEqual(imagen.read_bytes(), b"img")
            self.assertIsNone(etiqueta)                         # 404 en la API → sin etiqueta

    def test_la_api_no_deja_leer_fuera_de_crudos_y_exportados(self):
        r = self.api.get("/s3/url", params={"clave": "otros/secreto.txt"}, headers={"X-API-Key": "clave-s3"})
        self.assertEqual(r.status_code, 422)
        r = self.api.get("/s3/url", params={"clave": "crudos/../otros/secreto.txt"}, headers={"X-API-Key": "clave-s3"})
        self.assertEqual(r.status_code, 422)
        r = self.api.get("/s3/listar", params={"prefijo": "crudos/"}, headers={"X-API-Key": "otra"})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
