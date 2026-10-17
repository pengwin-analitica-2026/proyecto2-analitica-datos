import io
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs" / "s10" / "h8_aumm_noise01_noema_w32_e60_s42.pth"

try:
    from fastapi.testclient import TestClient
except ImportError:                                  # fastapi/httpx son opcionales para el resto del repo
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi/httpx no instalados")
@unittest.skipUnless(CHECKPOINT.exists(), "falta el checkpoint del candidato")
class ApiServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["API_KEY"] = "clave-de-prueba"
        os.environ["MODEL_PATH"] = str(CHECKPOINT)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "src"))
        sys.modules.pop("api.server", None)
        from api import server
        cls.client = TestClient(server.app)

    @staticmethod
    def npy(array):
        buffer = io.BytesIO()
        np.save(buffer, array)
        return buffer.getvalue()

    def test_health_reports_the_model(self):
        body = self.client.get("/health?deep=1").json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["model"], CHECKPOINT.name)
        self.assertTrue(body["canonical"])          # H8 espera cortes enderezados
        self.assertEqual(body["precision"], "int8")  # PRECISION=auto en CPU → int8 híbrido

    def test_wrong_key_is_rejected(self):
        response = self.client.post("/predict_slice", content=self.npy(np.zeros((384, 384), np.uint8)),
                                    headers={"X-API-Key": "otra"})
        self.assertEqual(response.status_code, 401)

    def test_bad_body_and_bad_shape(self):
        headers = {"X-API-Key": "clave-de-prueba"}
        self.assertEqual(self.client.post("/predict_slice", content=b"basura", headers=headers).status_code, 400)
        small = self.npy(np.zeros((10, 10), np.uint8))
        self.assertEqual(self.client.post("/predict_slice", content=small, headers=headers).status_code, 422)

    def test_valid_slice_returns_semantic_and_latency(self):
        import base64
        response = self.client.post("/predict_slice", content=self.npy(np.zeros((384, 384), np.uint8)),
                                    headers={"X-API-Key": "clave-de-prueba"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        with np.load(io.BytesIO(base64.b64decode(body["semantic_npz_b64"]))) as packed:
            self.assertEqual(packed["semantic"].shape, (384, 384))
        self.assertGreater(body["latency_ms"], 0)

    def test_batch_returns_one_result_per_slice(self):
        import base64
        images = np.zeros((3, 384, 384), np.uint8)
        response = self.client.post("/predict_batch", content=self.npy(images),
                                    headers={"X-API-Key": "clave-de-prueba"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        with np.load(io.BytesIO(base64.b64decode(body["arrays_npz_b64"]))) as packed:
            self.assertEqual(packed["semantic"].shape, (3, 384, 384))
            self.assertEqual(packed["primary_secondary"].shape, (3, 384, 384))
            self.assertEqual(packed["classification"].shape, (3, 3))
        self.assertEqual(len(body["boxes_refined"]), 3)

    def test_batch_limits(self):
        headers = {"X-API-Key": "clave-de-prueba"}
        self.assertEqual(self.client.post("/predict_batch", content=self.npy(np.zeros((2, 384, 384), np.uint8)),
                                          headers={"X-API-Key": "otra"}).status_code, 401)
        self.assertEqual(self.client.post("/predict_batch", content=self.npy(np.zeros((2, 64, 64), np.uint8)),
                                          headers=headers).status_code, 422)
        too_many = self.npy(np.zeros((17, 384, 384), np.uint8))
        self.assertIn(self.client.post("/predict_batch", content=too_many, headers=headers).status_code, (413, 422))

    def test_remote_predictor_matches_local_prediction(self):
        """inferencia_remota contra esta misma API (sin red): mismo resultado que predict_volume local."""
        from unittest import mock
        import inferencia_remota
        from predict_volume import load_pipeline, predict_volume

        def pedir(url, key, ruta, cuerpo=None, timeout=150.0):
            if cuerpo is None:
                return self.client.get(ruta, headers={"X-API-Key": key}).json()
            response = self.client.post(ruta, content=cuerpo, headers={"X-API-Key": key})
            if response.status_code != 200:
                raise RuntimeError(response.text)
            return response.json()

        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, (11, 384, 384), dtype=np.uint8)
        with mock.patch.object(inferencia_remota, "_pedir", pedir):
            info, predecir = inferencia_remota.predictor_remoto("http://api", "clave-de-prueba", lote=4, paralelo=2)
            remote = predecir(images)
        self.assertEqual(info["checkpoint"], CHECKPOINT.name)
        self.assertTrue(info["canonical"])
        model, device, _ = load_pipeline(CHECKPOINT, precision=info["precision"])   # la misma que el servidor
        local = predict_volume(model, images, device, batch_size=4)
        np.testing.assert_array_equal(remote["semantic"], local["semantic"])
        np.testing.assert_array_equal(remote["primary_secondary"], local["primary_secondary"])
        np.testing.assert_allclose(remote["classification"], local["classification"], atol=1e-5)
        self.assertEqual(len(remote["boxes_refined"]), 11)


if __name__ == "__main__":
    unittest.main()
