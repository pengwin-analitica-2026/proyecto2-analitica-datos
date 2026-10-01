"""Pruebas del preprocesamiento de entrada al modelo (pengwin_io.window_hu).

Verifican las propiedades en las que se apoya la decision de la asesoria de
semana 9 (gamma = 1.5, sin CLAHE) y la garantia de pipeline identico en
entrenamiento e inferencia.
"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pengwin_io import (GAMMA, WINDOW_CENTER, WINDOW_WIDTH,  # noqa: E402
                        clean_bone_mask, window_hu)

LOW = WINDOW_CENTER - WINDOW_WIDTH / 2     # -500 HU
HIGH = WINDOW_CENTER + WINDOW_WIDTH / 2    # 1300 HU


class WindowHuTests(unittest.TestCase):
    def test_gamma_por_defecto_es_el_documentado(self) -> None:
        self.assertEqual(GAMMA, 1.5)
        self.assertEqual(inspect.signature(window_hu).parameters["gamma"].default, GAMMA)

    def test_rango_y_tipo(self) -> None:
        hu = np.array([[-3000, -1000, -500, 0, 250, 800, 1300, 5000]], dtype=np.float32)
        out = window_hu(hu)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        # Los extremos de la ventana quedan fijos en 0 y 1 con cualquier gamma.
        self.assertEqual(float(out[0, 2]), 0.0)
        self.assertEqual(float(out[0, 6]), 1.0)

    def test_gamma_uno_es_el_ventaneo_lineal(self) -> None:
        hu = np.linspace(-1200, 2000, 50, dtype=np.float32)[None, :]
        esperado = (np.clip(hu, LOW, HIGH) - LOW) / (HIGH - LOW)
        np.testing.assert_allclose(window_hu(hu, gamma=1.0), esperado, atol=1e-6)

    def test_monotona_no_destruye_el_orden(self) -> None:
        """Gamma redistribuye grises pero conserva el orden: HU mayor, gris mayor."""
        hu = np.linspace(LOW, HIGH, 200, dtype=np.float32)[None, :]
        out = window_hu(hu)[0]
        self.assertTrue(bool(np.all(np.diff(out) > 0)))

    def test_comprime_tejido_blando_y_expande_hueso(self) -> None:
        """La razon de usar gamma > 1 en una ventana osea."""
        blando = np.array([[0.0, 60.0]], dtype=np.float32)      # musculo / organos
        hueso = np.array([[700.0, 760.0]], dtype=np.float32)    # hueso trabecular-cortical
        sep = lambda img: float(img[0, 1] - img[0, 0])          # noqa: E731
        lin_b, lin_h = window_hu(blando, gamma=1.0), window_hu(hueso, gamma=1.0)
        g_b, g_h = window_hu(blando), window_hu(hueso)
        # Misma diferencia de 60 HU: en tejido blando se achica, en hueso crece.
        self.assertLess(sep(g_b), sep(lin_b))
        self.assertGreater(sep(g_h), sep(lin_h))

    def test_cada_voxel_depende_solo_de_su_hu(self) -> None:
        """Pipeline identico: el mismo HU da el mismo gris en cualquier corte.

        Con la renormalizacion por imagen del paper esto fallaria en un corte
        sin hueso denso. Es la razon por la que no se renormaliza.
        """
        con_hueso = np.array([[-1000.0, 40.0, 1500.0]], dtype=np.float32)
        sin_hueso = np.array([[-1000.0, 40.0, 300.0]], dtype=np.float32)
        self.assertEqual(float(window_hu(con_hueso)[0, 1]), float(window_hu(sin_hueso)[0, 1]))

    def test_mismo_resultado_en_2d_y_3d(self) -> None:
        rng = np.random.default_rng(0)
        vol = rng.uniform(-1200, 2500, size=(3, 16, 16)).astype(np.float32)
        por_corte = np.stack([window_hu(c) for c in vol])
        np.testing.assert_array_equal(window_hu(vol), por_corte)

    def test_gamma_invalido(self) -> None:
        with self.assertRaises(ValueError):
            window_hu(np.zeros((2, 2), dtype=np.float32), gamma=0.0)


class CleanBoneMaskDefaultsTests(unittest.TestCase):
    def test_histeresis_desactivada_por_defecto(self) -> None:
        """Asesoria semana 9: la recuperacion 150-250 HU con cierre no corre por defecto."""
        params = inspect.signature(clean_bone_mask).parameters
        self.assertIsNone(params["low_threshold"].default)
        self.assertFalse(params["fill_cavities"].default)


if __name__ == "__main__":
    unittest.main()