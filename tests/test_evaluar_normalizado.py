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


class TestEstresEnMilimetros(unittest.TestCase):
    def _pelvis_en_pixeles(self, mm_x, mm_y):
        recta = _pelvis_sintetica(0, n=240)
        from scipy import ndimage as ndi
        s = (ndi.zoom(recta, (1 / mm_y, 1 / mm_x), order=0) > 0.5).astype(np.uint8)
        return (s * 200)[None].repeat(2, axis=0), (s * 11)[None].repeat(2, axis=0)

    def test_rotmm_igual_a_rot_con_pixeles_cuadrados(self):
        img, lab = self._pelvis_en_pixeles(1.0, 1.0)
        a = apply_stress(img, lab, "rot=40")
        b = apply_stress(img, lab, "rotmm=40", (0.8, 0.8))
        np.testing.assert_array_equal(a[1], b[1])

    def test_rotmm_se_recupera_con_pixeles_no_cuadrados(self):
        """Píxeles de 0.84 × 0.42 mm (como el caso 044): el giro físico sí se recupera."""
        mm_xy = (0.84, 0.42)
        img, lab = self._pelvis_en_pixeles(*mm_xy)
        for rot in (30, 60):
            with self.subTest(rot=rot):
                img_g, lab_g = apply_stress(img, lab, f"rotmm={rot}", mm_xy)
                _, lab_e, est = nz.enderezar_caso(img_g, lab_g, mm_xy)
                self.assertAlmostEqual(est.angulo, rot, delta=2.0)
                self.assertGreater(_dice(lab_e[0], lab[0]), 0.93)


if __name__ == "__main__":
    unittest.main()
