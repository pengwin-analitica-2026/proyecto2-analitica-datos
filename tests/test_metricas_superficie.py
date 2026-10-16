import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metricas_superficie import fragment_rows, summarize, summarize_groups


def cubo(volumen: np.ndarray, valor: int, z: int, y: int, x: int, lado: int = 6) -> None:
    volumen[z:z + lado, y:y + lado, x:x + lado] = valor


class FragmentRowsTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.zeros((20, 40, 40), dtype=np.uint8)
        cubo(self.truth, 1, 2, 2, 2)      # sacro principal
        cubo(self.truth, 2, 2, 20, 20)    # sacro secundario
        cubo(self.truth, 11, 12, 2, 20)   # coxal izquierdo principal

    def test_prediccion_perfecta_da_cero_mm_y_cobertura_total(self):
        rows, extra = fragment_rows(self.truth.copy(), self.truth, (1.0, 1.0, 1.0))
        self.assertEqual(extra, 0)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(row["detectado"])
            self.assertAlmostEqual(row["hd95_mm"], 0.0)
            self.assertAlmostEqual(row["assd_mm"], 0.0)
        self.assertEqual(summarize(rows)["cobertura"], 1.0)

    def test_desplazamiento_se_mide_en_milimetros(self):
        pred = np.zeros_like(self.truth)
        cubo(pred, 1, 2, 2, 4)            # el principal del sacro, corrido 2 vóxeles en x
        rows, _ = fragment_rows(pred, self.truth, (0.5, 1.0, 1.0))   # x mide 0.5 mm
        sacro = next(r for r in rows if r["gt_id"] == 1)
        self.assertAlmostEqual(sacro["hd95_mm"], 1.0)   # 2 vóxeles x 0.5 mm
        self.assertGreater(sacro["assd_mm"], 0.0)
        self.assertLess(sacro["assd_mm"], 1.0)

    def test_fragmento_sin_pareja_cuenta_como_no_detectado(self):
        pred = self.truth.copy()
        pred[pred == 2] = 0               # se pierde el secundario del sacro
        cubo(pred, 21, 12, 30, 30)        # y sobra un coxal derecho que no existe
        rows, extra = fragment_rows(pred, self.truth, (1.0, 1.0, 1.0))
        perdido = next(r for r in rows if r["gt_id"] == 2)
        self.assertFalse(perdido["detectado"])
        self.assertIsNone(perdido["hd95_mm"])
        self.assertEqual(extra, 1)
        grupos = summarize_groups(rows)
        self.assertEqual(grupos["todos"]["n_no_detectados"], 1)
        self.assertAlmostEqual(grupos["todos"]["cobertura"], 2 / 3)
        self.assertEqual(grupos["secundario"]["cobertura"], 0.0)
        self.assertEqual(grupos["principal"]["cobertura"], 1.0)
        self.assertNotIn("hd95_mm_media", grupos["secundario"])

    def test_un_predicho_no_se_usa_dos_veces(self):
        pred = np.zeros_like(self.truth)
        pred[2:8, 2:26, 2:26] = 1         # un solo fragmento que cubre los dos del sacro
        rows, _ = fragment_rows(pred, self.truth, (1.0, 1.0, 1.0))
        detectados = [r for r in rows if r["region"] == "sacro" and r["detectado"]]
        self.assertEqual(len(detectados), 1)


if __name__ == "__main__":
    unittest.main()
