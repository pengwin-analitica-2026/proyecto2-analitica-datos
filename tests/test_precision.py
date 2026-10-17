"""H8 en las tres precisiones: fp32, int8 híbrido (CPU) y fp16; misma predicción, int8 listo en archivo."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import precision  # noqa: E402

CHECKPOINT = ROOT / "outputs" / "s10" / "h8_aumm_noise01_noema_w32_e60_s42.pth"


class Elegir(unittest.TestCase):
    def test_auto_segun_el_dispositivo(self):
        self.assertEqual(precision.elegir("auto", torch.device("cpu")), "int8")
        self.assertEqual(precision.elegir("auto", torch.device("cuda")), "fp16")
        self.assertEqual(precision.elegir("FP32", torch.device("cpu")), "fp32")
        with self.assertRaises(ValueError):
            precision.elegir("int8", torch.device("cuda"))        # int8 de PyTorch solo existe en CPU
        with self.assertRaises(ValueError):
            precision.elegir("int4", torch.device("cpu"))


@unittest.skipUnless(CHECKPOINT.exists() and precision.CALIBRACION.exists(), "falta el checkpoint o la calibración")
class TresPrecisiones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from predict_volume import load_pipeline, predict_volume
        cls.load_pipeline, cls.predict_volume = staticmethod(load_pipeline), staticmethod(predict_volume)
        with np.load(precision.CALIBRACION) as datos:
            cls.cortes = datos["cortes"][::16]                    # 8 cortes de entrenamiento con hueso
        cls.fp32 = cls.predecir("fp32")

    @classmethod
    def predecir(cls, p):
        model, device, info = cls.load_pipeline(CHECKPOINT, torch.device("cpu"), tta=False, precision=p)
        return info, cls.predict_volume(model, cls.cortes, device, batch_size=4)

    def test_int8_predice_lo_mismo_que_fp32(self):
        info, pred = self.predecir("int8")
        self.assertEqual(info["precision"], "int8")
        self.assertTrue(info["canonical"])
        acuerdo = float((pred["semantic"] == self.fp32[1]["semantic"]).mean())
        self.assertGreater(acuerdo, 0.995)
        self.assertGreater(float((pred["primary_secondary"] == self.fp32[1]["primary_secondary"]).mean()), 0.995)

    def test_int8_exportado_se_carga_igual(self):
        with tempfile.TemporaryDirectory() as d:
            copia = Path(d) / CHECKPOINT.name
            copia.write_bytes(CHECKPOINT.read_bytes())
            salida = precision.exportar_int8(copia)
            self.assertEqual(salida, precision.ruta_int8(copia))
            self.assertLess(salida.stat().st_size, 6e6)          # menos de la mitad que fp32 (11 MB)
            model, device, info = self.load_pipeline(copia, torch.device("cpu"), precision="int8")
            self.assertTrue(info["precision_origen"].startswith("archivo"))
            cargado = self.predict_volume(model, self.cortes, device, batch_size=4)
            model, device, _ = self.load_pipeline(copia, torch.device("cpu"), precision="int8")
            np.testing.assert_array_equal(cargado["semantic"], self.predict_volume(
                model, self.cortes, device, batch_size=4)["semantic"])

    def test_auto_en_cpu_es_int8_y_con_tta(self):
        model, device, info = self.load_pipeline(CHECKPOINT, torch.device("cpu"), tta=True, precision="auto")
        self.assertEqual(info["precision"], "int8")
        self.assertTrue(info["tta"])
        pred = self.predict_volume(model, self.cortes[:2], device, batch_size=2)
        self.assertEqual(pred["semantic"].shape, (2, 384, 384))


if __name__ == "__main__":
    unittest.main()
