import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paquete_web import empaquetar, escritor_local, resumen_proyecto  # noqa: E402


def exportacion_falsa(carpeta: Path, con_gt: bool = True) -> None:
    """Una carpeta como la de demo_export, chica: 3 cortes de 8×8 y un fragmento."""
    images = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    ids = np.zeros((3, 8, 8), np.uint8)
    ids[1, 2:5, 2:5] = 1
    arrays = {"images": images, "semantic": (ids > 0).astype(np.uint8), "ids": ids}
    if con_gt:
        arrays["gt_ids"] = ids
    np.savez_compressed(carpeta / "volume.npz", **arrays)
    np.savez_compressed(carpeta / "meshes.npz", p_1_v=np.ones((4, 3), np.float32) * 2.5,
                        p_1_f=np.array([[0, 1, 2], [1, 2, 3]], np.int32))
    np.savez_compressed(carpeta / "raw_view.npz", mip_axial=images[0], mip_coronal=images[1], mip_sagital=images[2],
                        bone_verts_zyx=np.zeros((3, 3), np.float32), bone_faces=np.array([[0, 1, 2]], np.int32))
    np.savez_compressed(carpeta / "pasos.npz", crudo_corte=images[0], crudo_mip=images[1])
    meta = {"case_id": "x", "n_slices": 3, "size": 8, "pasos": [{"clave": "crudo", "titulo": "t", "texto": "", "z": 1}]}
    (carpeta / "meta.json").write_text(json.dumps(meta))


class PaqueteWeb(unittest.TestCase):
    def test_empaqueta_todo_y_caso_json_al_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "exp").mkdir()
            exportacion_falsa(tmp / "exp")
            orden = []
            local = escritor_local(tmp / "web")

            def escribir(nombre, datos):
                orden.append(nombre)
                local(nombre, datos)
            caso = empaquetar(tmp / "exp", escribir)
            self.assertEqual(orden[-1], "caso.json")
            p = caso["paquete"]
            self.assertEqual(sorted(n for n in orden if n.startswith("cortes/")), ["cortes/0000.png", "cortes/0001.png", "cortes/0002.png"])
            ids = np.frombuffer(gzip.decompress((tmp / "web" / p["ids"]).read_bytes()), np.uint8).reshape(3, 8, 8)
            self.assertEqual(int(ids[1].sum()), 9)
            self.assertEqual(p["gt"], "gt.bin.gz")
            # Las mallas se leen con los desplazamientos del índice, como en el navegador.
            datos = gzip.decompress((tmp / "web" / "mallas.bin.gz").read_bytes())
            item = p["mallas"]["items"][0]
            self.assertEqual(item["clave"], "p_1")
            verts = np.frombuffer(datos, np.float32, count=item["v"][1], offset=item["v"][0])
            faces = np.frombuffer(datos, np.uint32, count=item["f"][1], offset=item["f"][0])
            self.assertTrue(np.allclose(verts, 2.5))
            self.assertEqual(faces.tolist(), [0, 1, 2, 1, 2, 3])
            self.assertEqual(set(p["mips"]), {"axial", "coronal", "sagital"})
            self.assertIn("crudo", p["hueso"])
            self.assertEqual(p["pasos"], {"crudo_corte": "pasos/crudo_corte.png", "crudo_mip": "pasos/crudo_mip.png"})

    def test_los_cortes_png_conservan_los_pixeles(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "exp").mkdir()
            exportacion_falsa(tmp / "exp", con_gt=False)
            caso = empaquetar(tmp / "exp", escritor_local(tmp / "web"))
            self.assertIsNone(caso["paquete"]["gt"])
            corte = np.asarray(Image.open(tmp / "web" / "cortes" / "0002.png"))
            self.assertTrue(np.array_equal(corte, np.arange(3 * 64, dtype=np.uint8).reshape(3, 8, 8)[2]))

    def test_resumen_con_los_json_del_repo(self):
        resumen = resumen_proyecto(ROOT / "outputs", ROOT / "outputs" / "demo")
        if resumen["test"] is None:
            self.skipTest("faltan los JSON de test en outputs/")
        self.assertEqual(len(resumen["test"]["filas"]), 9)
        self.assertIn("4.ª · H8", resumen["test"]["columnas"])


try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi/httpx no instalados")
class ServidorWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ.pop("WEB_BUCKET", None)
        os.environ.pop("INFERENCE_URL", None)
        os.environ["WEB_DATA_DIR"] = cls.tmp.name
        sys.path.insert(0, str(ROOT))
        sys.modules.pop("web.servidor", None)
        from web import servidor
        cls.servidor = servidor
        cls.client = TestClient(servidor.app)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_config_solo_cpu_sin_gpu(self):
        datos = self.client.get("/api/config").json()
        self.assertEqual([d["id"] for d in datos["destinos"]], ["cpu"])
        self.assertIsNone(datos["gpu"])

    def test_rechaza_casos_y_destinos_raros(self):
        self.assertEqual(self.client.post("/api/abrir", json={"caso": "../x", "modo": "precalculado"}).status_code, 422)
        self.assertEqual(self.client.post("/api/abrir", json={"caso": "001", "modo": "vivo", "destino": "tpu"}).status_code, 422)
        self.assertEqual(self.client.get("/api/trabajos/no-existe").status_code, 404)

    def test_precalculado_listo_no_crea_trabajo(self):
        prefijo = f"resultados/{self.servidor.MODELO}/pengwin/007"
        self.servidor.ALMACEN.escribir(f"{prefijo}/caso.json", b"{}")
        self.assertEqual(self.client.post("/api/abrir", json={"caso": "007", "modo": "precalculado"}).json(),
                         {"resultado": prefijo})

    def test_subida_guarda_los_archivos_con_nombre_seguro(self):
        r = self.client.post("/api/subidas", json={"archivos": [{"nombre": "serie/../CT 01.dcm", "tamano": 3}]}).json()
        formulario = r["formularios"][0]
        self.assertTrue(formulario["fields"]["key"].endswith("/00000_CT_01.dcm"))
        enviado = self.client.post("/api/dev/subir", data=formulario["fields"], files={"file": ("CT 01.dcm", b"abc")})
        self.assertEqual(enviado.status_code, 204)
        self.assertEqual(self.servidor.ALMACEN.leer(formulario["fields"]["key"]), b"abc")
        demasiado = self.client.post("/api/subidas", json={"archivos": [{"nombre": "a.mha", "tamano": 10 ** 12}]})
        self.assertEqual(demasiado.status_code, 413)

    def test_corte_exige_384x384_bytes(self):
        self.assertEqual(self.client.post("/api/corte", content=b"\x00" * 10).status_code, 422)


if __name__ == "__main__":
    unittest.main()
