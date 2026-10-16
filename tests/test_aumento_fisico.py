"""Pruebas de src/aumento_fisico.py: el giro de entrenamiento en milímetros reales."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aumento_fisico as af  # noqa: E402


def _elipse_de_un_circulo(n=128, radio_mm=40.0, mm_x=1.0, mm_y=2.0):
    """Un círculo físico de ``radio_mm`` dibujado en píxeles no cuadrados: se ve como elipse."""
    yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    c = (n - 1) / 2
    dentro = ((xx - c) * mm_x) ** 2 + ((yy - c) * mm_y) ** 2 < radio_mm ** 2
    return dentro.float()[None, None], dentro.long()[None] * 11


def _dice(a, b):
    a, b = a.bool(), b.bool()
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def _girar(x, y, grados, mm_xy):
    cero = torch.zeros(1)
    theta = af.theta_mm(torch.tensor([math.radians(grados)]), torch.ones(1), torch.ones(1),
                        torch.zeros(1, 2), mm_xy, x.shape[-2], x.shape[-1])
    return af.aplicar_afin(x, y, theta)


class TestGiroEnMilimetros(unittest.TestCase):
    def test_circulo_fisico_sigue_siendo_circulo(self):
        """Girar 90° un círculo real no lo cambia; en píxeles no cuadrados sigue siendo la misma elipse."""
        x, y = _elipse_de_un_circulo(mm_x=1.0, mm_y=2.0)
        _, y_mm = _girar(x, y, 90, torch.tensor([[1.0, 2.0]]))
        self.assertGreater(_dice(y_mm[0], y[0]), 0.95)

    def test_giro_en_pixeles_deforma(self):
        """El método anterior (en píxeles) convierte la elipse horizontal en vertical: deforma la anatomía."""
        x, y = _elipse_de_un_circulo(mm_x=1.0, mm_y=2.0)
        _, y_px = _girar(x, y, 90, None)
        self.assertLess(_dice(y_px[0], y[0]), 0.80)

    def test_pixeles_cuadrados_igual_que_antes(self):
        """Con mm iguales en x e y, el resultado es idéntico al método anterior (reproducibilidad)."""
        x = torch.rand(4, 1, 64, 64)
        y = (torch.rand(4, 64, 64) * 30).long()
        torch.manual_seed(7); xa, ya = af.random_affine_mm(x, y, None, probability=1.0, rotation_degrees=35)
        torch.manual_seed(7); xb, yb = af.random_affine_mm(x, y, torch.full((4, 2), 0.8),
                                                           probability=1.0, rotation_degrees=35)
        self.assertTrue(torch.allclose(xa, xb, atol=1e-5))
        self.assertTrue(torch.equal(ya, yb))

    def test_etiqueta_sin_ids_nuevos(self):
        x = torch.rand(3, 1, 64, 64)
        y = torch.zeros(3, 64, 64, dtype=torch.long)
        y[:, 10:30, 10:30] = 12
        y[:, 35:55, 30:50] = 21
        _, y2 = af.random_affine_mm(x, y, torch.tensor([[0.7, 1.4]] * 3), probability=1.0,
                                    rotation_degrees=35)
        self.assertTrue(set(torch.unique(y2).tolist()) <= {0, 12, 21})

    def test_mm_por_pixel(self):
        caso = {"spacing_xyz": np.array([0.8, 0.8, 1.0]), "orig_shape": np.array([200, 300, 512]),
                "size": np.array(384), "mode": np.array("stretch")}
        mx, my = af.mm_por_pixel(caso)
        self.assertAlmostEqual(mx, 0.8 * 512 / 384)
        self.assertAlmostEqual(my, 0.8 * 300 / 384)
        caso["mode"] = np.array("pad")
        mx, my = af.mm_por_pixel(caso)
        self.assertAlmostEqual(mx, my)


if __name__ == "__main__":
    unittest.main()
