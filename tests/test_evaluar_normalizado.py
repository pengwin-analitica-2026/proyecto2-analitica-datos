"""El estrés de giro de evaluate_volumes y el enderezado de normalizacion.py encajan.

Simula lo que hace ``evaluate_volumes.py --stress rot=X --normalizar`` con una pelvis sintética:
la prueba de estrés gira el volumen y el paso 2 debe estimar el mismo ángulo (mismo signo) y
dejar la etiqueta casi igual a la original.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

import normalizacion as nz  # noqa: E402
from evaluate_volumes import apply_stress  # noqa: E402
from test_normalizacion import _pelvis_sintetica  # noqa: E402


def _dice(a, b):
    a, b = a > 0, b > 0
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


class TestEstresMasEnderezado(unittest.TestCase):
    def test_recupera_el_giro_del_estres(self):
        recta = _pelvis_sintetica(0, n=200)
        imagenes = (recta * 200).astype(np.uint8)[None].repeat(3, axis=0)
        etiquetas = (recta * 11).astype(np.uint8)[None].repeat(3, axis=0)
        for rot in (-30, 30, 60):
            with self.subTest(rot=rot):
                img_g, lab_g = apply_stress(imagenes, etiquetas, f"rot={rot}")
                img_e, lab_e, est = nz.enderezar_caso(img_g, lab_g, (0.8, 0.8))
                self.assertAlmostEqual(est.angulo, rot, delta=1.5)
                self.assertTrue(est.corregir)
                self.assertLess(_dice(lab_g[0], etiquetas[0]), 0.9)
                self.assertGreater(_dice(lab_e[0], etiquetas[0]), 0.97)
                self.assertTrue(set(np.unique(lab_e)) <= {0, 11})


if __name__ == "__main__":
    unittest.main()
