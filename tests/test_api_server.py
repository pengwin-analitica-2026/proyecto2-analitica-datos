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
        self.assertTrue(body["canonical"])          # H6 espera cortes enderezados

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


if __name__ == "__main__":
    unittest.main()
