import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import resumen_s10  # noqa: E402


class ResumenS10Test(unittest.TestCase):
    def test_tabla_objetivos_lists_the_nine_course_metrics(self) -> None:
        tabla = resumen_s10.tabla_objetivos([("H6", "TEST2_h6_s42_tta"), ("D5", "TEST_d5_s42_tta")])

        self.assertEqual(len(tabla), 9)
        self.assertTrue((tabla["H6 cumple"] == "sí").all())
        self.assertEqual(tabla.loc["Dice por fragmento", "D5 cumple"], "NO")
        self.assertAlmostEqual(tabla.loc["Dice por fragmento", "H6"], 0.875, places=3)

    def test_tabla_modelos_keeps_missing_runs_as_empty_rows(self) -> None:
        tabla = resumen_s10.tabla_modelos([("existe", "d5_ema_w32_e60_s42_val"), ("falta", "no_existe_val")])

        self.assertAlmostEqual(tabla.loc["existe", "Dice inst."], 0.851, places=3)
        self.assertTrue(tabla.loc["falta"].isna().all())

    def test_tabla_por_caso_has_one_row_per_test_case(self) -> None:
        tabla = resumen_s10.tabla_por_caso([("D5", "TEST_d5_s42_tta"), ("H6", "TEST2_h6_s42_tta")])

        self.assertEqual(len(tabla), 15)
        self.assertGreater(tabla.loc["012", "H6"], tabla.loc["012", "D5"])


class InformeTestTest(unittest.TestCase):
    def test_classification_confusion_counts_each_outcome(self) -> None:
        from informe_test import confusion_clasificacion

        def record(logits, target):
            return SimpleNamespace(cls_logits=np.asarray(logits, dtype=np.float32),
                                   cls_target=np.asarray(target, dtype=np.float32))

        records = [record([2, -2, 2], [1, 0, 1]), record([2, 2, -2], [1, 0, 1]), record([-2, -2, -2], [1, 0, 0])]
        out = confusion_clasificacion(records)

        self.assertEqual(out["sacro"]["presente_y_predicha"], 2)
        self.assertEqual(out["sacro"]["presente_y_no_predicha"], 1)
        self.assertEqual(out["coxal izquierdo"]["ausente_y_predicha"], 1)
        self.assertEqual(out["coxal izquierdo"]["ausente_y_no_predicha"], 2)
        self.assertEqual(out["coxal derecho"]["presente_y_predicha"], 1)
        self.assertAlmostEqual(out["sacro"]["f1"], 0.8, places=4)

    def test_secondary_fragments_use_a_lighter_tone_of_the_region(self) -> None:
        from informe_test import colorear

        rgba = colorear(np.array([[0, 11, 12]]))

        self.assertEqual(rgba[0, 0, 3], 0.0)
        self.assertGreater(rgba[0, 2, :3].sum(), rgba[0, 1, :3].sum())


if __name__ == "__main__":
    unittest.main()
