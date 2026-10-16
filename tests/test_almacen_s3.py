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


class AlmacenS3(unittest.TestCase):
    def setUp(self):
        self.s3 = S3Falso({"crudos/imagenes/024.mha": b"img", "crudos/etiquetas/024.mha": b"lbl",
                           "crudos/imagenes/044.mha": b"img2",
                           "exportados/h8/044/meta.json": b"{}", "exportados/h8/044/volume.npz": b"v"})
        parche = mock.patch.object(almacen_s3, "cliente", lambda: self.s3)
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


if __name__ == "__main__":
    unittest.main()
