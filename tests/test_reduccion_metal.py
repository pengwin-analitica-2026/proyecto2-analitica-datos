"""Reducción de artefactos de metal (LI-MAR): no toca cortes sin metal, repone el metal y atenúa rayas."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reduccion_metal import li_mar, reducir_artefactos_metal  # noqa: E402


def fantoma(con_metal=True, rayas=True, n=128):
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    corte = np.full((n, n), -1024.0, dtype=np.float32)
    cuerpo = (yy - n / 2) ** 2 / (0.4 * n) ** 2 + (xx - n / 2) ** 2 / (0.45 * n) ** 2 <= 1
    corte[cuerpo] = 40.0
    cy, cx = n / 2, n / 2 - 22
    metal = (yy - cy) ** 2 + (xx - cx) ** 2 <= 16
    if rayas:   # rayas radiales que salen del metal, como las del archivo G3_P1_A del profesor
        angulo = np.arctan2(yy - cy, xx - cx)
        corte[cuerpo] += 250 * np.cos(16 * angulo[cuerpo])
    if con_metal:
        corte[metal] = 3000.0
    return corte, cuerpo & ~metal


class TestLiMar(unittest.TestCase):
    def test_sin_metal_no_cambia(self):
        corte, _ = fantoma(con_metal=False)
        np.testing.assert_array_equal(li_mar(corte, n_angulos=90), corte)

    def test_repone_metal_y_aire(self):
        corte, _ = fantoma()
        salida = li_mar(corte, n_angulos=90)
        np.testing.assert_array_equal(salida[corte >= 2500], corte[corte >= 2500])
        np.testing.assert_array_equal(salida[corte <= -1024], corte[corte <= -1024])
        self.assertTrue(np.isfinite(salida).all())

    def test_atenua_las_rayas(self):
        corte, tejido = fantoma()
        salida = li_mar(corte, n_angulos=180)
        antes, despues = np.std(corte[tejido]), np.std(salida[tejido])
        self.assertLess(despues, 0.6 * antes)

    def test_volumen_solo_cortes_con_metal(self):
        con, _ = fantoma()
        sin, _ = fantoma(con_metal=False)
        volumen = np.stack([sin, con, sin])
        salida, resumen = reducir_artefactos_metal(volumen, hilos=2, log=lambda *_: None)
        self.assertEqual(resumen["cortes"], [1])
        np.testing.assert_array_equal(salida[0], sin)
        np.testing.assert_array_equal(salida[2], sin)
        self.assertFalse(np.array_equal(salida[1], con))


if __name__ == "__main__":
    unittest.main()
