"""Pruebas de src/normalizacion.py (paso 1: metadatos; paso 2: giro axial).

Las pruebas con volúmenes sintéticos corren siempre. La comparación con los 100 casos reales solo
corre si existe data/raw (como test_orientacion.py).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import normalizacion as nz  # noqa: E402


def _imagen(arreglo, direccion=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(0.8, 0.9, 1.25)):
    img = sitk.GetImageFromArray(arreglo)
    img.SetDirection(tuple(float(v) for v in direccion))
    img.SetSpacing(spacing)
    return img


def _pelvis_sintetica(angulo=0.0, n=160):
    """Silueta axial simétrica: dos coxales, sacro atrás y pubis adelante (asimétrica adelante/atrás)."""
    yy, xx = np.mgrid[:n, :n]
    c = n / 2
    s = np.zeros((n, n), dtype=np.float32)
    s[((xx - c + 38) / 16) ** 2 + ((yy - c) / 34) ** 2 < 1] = 1      # coxal
    s[((xx - c - 38) / 16) ** 2 + ((yy - c) / 34) ** 2 < 1] = 1      # coxal
    s[((xx - c) / 20) ** 2 + ((yy - c - 30) / 12) ** 2 < 1] = 1      # sacro (atrás)
    s[((xx - c) / 10) ** 2 + ((yy - c + 40) / 6) ** 2 < 1] = 1       # pubis (adelante)
    if angulo:
        s = (ndi.rotate(s, angulo, reshape=False, order=0) > 0.5).astype(np.float32)
    return s


class TestPaso1Metadatos(unittest.TestCase):
    def test_diagnostico_identidad(self):
        d = nz.diagnosticar(np.eye(3).ravel())
        self.assertTrue(d.ortogonal and d.es_estandar and not d.espejado and not d.oblicuo)

    def test_diagnostico_lpi_giro_180(self):
        d = nz.diagnosticar((-1, 0, 0, 0, -1, 0, 0, 0, 1))
        self.assertTrue(d.ortogonal)
        self.assertFalse(d.oblicuo)
        self.assertFalse(d.espejado)                # girar 180° no es un espejo
        self.assertEqual(d.signos, (-1, -1, 1))

    def test_diagnostico_espejado(self):
        d = nz.diagnosticar((1, 0, 0, 0, 1, 0, 0, 0, -1))
        self.assertTrue(d.espejado)

    def test_diagnostico_oblicuo_y_no_ortogonal(self):
        a = np.radians(20)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        self.assertTrue(nz.diagnosticar(rot.ravel()).oblicuo)
        self.assertFalse(nz.diagnosticar((1, 0.5, 0, 0, 1, 0, 0, 0, 1)).ortogonal)

    def test_lpi_igual_que_to_canonical(self):
        """Los 34 casos LPI: el método general da lo mismo que voltear y y x (to_canonical)."""
        arr = np.random.default_rng(0).integers(-1000, 2000, (5, 7, 9)).astype(np.int16)
        img = _imagen(arr, (-1, 0, 0, 0, -1, 0, 0, 0, 1))
        out, _, diag = nz.a_orientacion_estandar(img)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(out), arr[:, ::-1, ::-1])
        self.assertTrue(nz.diagnosticar(out.GetDirection()).es_estandar)

    def test_rai_no_cambia(self):
        arr = np.arange(5 * 7 * 9, dtype=np.int16).reshape(5, 7, 9)
        out, _, diag = nz.a_orientacion_estandar(_imagen(arr))
        self.assertTrue(diag.es_estandar)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(out), arr)

    def test_ejes_permutados_conservan_spacing(self):
        """Ejes x e y intercambiados (cabecera tipo NIfTI): se reordenan y el spacing los sigue."""
        arr = np.arange(5 * 7 * 9, dtype=np.int16).reshape(5, 7, 9)
        img = _imagen(arr, (0, 1, 0, 1, 0, 0, 0, 0, 1), spacing=(0.8, 0.9, 1.25))
        out, _, _ = nz.a_orientacion_estandar(img)
        self.assertEqual(sitk.GetArrayFromImage(out).shape, (5, 9, 7))
        np.testing.assert_allclose(out.GetSpacing(), (0.9, 0.8, 1.25))
        self.assertTrue(nz.diagnosticar(out.GetDirection()).es_estandar)

    def test_etiqueta_igual_transformacion_sin_ids_nuevos(self):
        arr = np.zeros((4, 6, 8), dtype=np.uint8)
        arr[:, 1:3, 2:5] = 11
        arr[:, 4:6, 5:8] = 23
        lbl = _imagen(arr, (-1, 0, 0, 0, -1, 0, 0, 0, 1))
        img = _imagen(arr.astype(np.int16), (-1, 0, 0, 0, -1, 0, 0, 0, 1))
        out_img, out_lbl, _ = nz.a_orientacion_estandar(img, lbl)
        a, b = sitk.GetArrayFromImage(out_img), sitk.GetArrayFromImage(out_lbl)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(set(np.unique(b)), {0, 11, 23})

    def test_oblicuo_se_remuestrea_y_etiqueta_sin_ids_nuevos(self):
        a = np.radians(15)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        lab = np.zeros((6, 30, 30), dtype=np.uint8)
        lab[:, 10:20, 5:12] = 11
        lab[:, 10:20, 18:25] = 21
        img = _imagen(lab.astype(np.int16) * 50, rot.ravel(), spacing=(1.0, 1.0, 1.0))
        lbl = _imagen(lab, rot.ravel(), spacing=(1.0, 1.0, 1.0))
        out_img, out_lbl, diag = nz.a_orientacion_estandar(img, lbl)
        self.assertTrue(diag.oblicuo)
        self.assertTrue(nz.diagnosticar(out_img.GetDirection()).es_estandar)
        self.assertEqual(out_img.GetSize(), out_lbl.GetSize())
        self.assertTrue(set(np.unique(sitk.GetArrayFromImage(out_lbl))) <= {0, 11, 21})


class TestPaso2GiroAxial(unittest.TestCase):
    def test_recupera_angulos_conocidos(self):
        for real in (-60, -30, -15, 10, 26, 45, 70):
            est = nz.estimar_giro_axial(_pelvis_sintetica(real))
            with self.subTest(angulo=real):
                self.assertAlmostEqual(est.angulo, real, delta=1.5)
                self.assertTrue(est.corregir)

    def test_no_confunde_con_90_grados(self):
        """Silueta casi simétrica también adelante-atrás: no debe elegir el ángulo a 90°."""
        n = 160; yy, xx = np.mgrid[:n, :n]; c = n / 2
        s = np.zeros((n, n), dtype=np.float32)
        s[((xx - c) / 60) ** 2 + ((yy - c) / 45) ** 2 < 1] = 1            # más ancha que profunda
        s[((xx - c) / 8) ** 2 + ((yy - c - 30) / 8) ** 2 < 1] = 0          # hueco atrás (asimetría leve)
        for real in (-60, -20, 30, 60):
            with self.subTest(angulo=real):
                girada = (ndi.rotate(s, real, reshape=False, order=0) > 0.5).astype(np.float32)
                est = nz.estimar_giro_axial(girada)
                self.assertAlmostEqual(est.angulo, real, delta=2.0)

    def test_paciente_recto_no_se_corrige(self):
        est = nz.estimar_giro_axial(_pelvis_sintetica(0))
        self.assertLess(abs(est.angulo), 1.5)
        self.assertFalse(est.corregir)

    def test_giro_pequeno_no_se_corrige(self):
        self.assertFalse(nz.estimar_giro_axial(_pelvis_sintetica(3)).corregir)

    def test_corregir_endereza_y_etiqueta_conserva_ids(self):
        recta = _pelvis_sintetica(0)
        etiqueta = (recta * 11).astype(np.uint8)[None].repeat(3, axis=0)
        girada = (ndi.rotate(etiqueta, 20, axes=(1, 2), reshape=False, order=0))
        est = nz.estimar_giro_axial((girada[0] > 0).astype(np.float32))
        enderezada = nz.corregir_giro_axial(girada, est.angulo, es_etiqueta=True)
        self.assertEqual(set(np.unique(enderezada)), {0, 11})
        inter = np.logical_and(enderezada[0] > 0, etiqueta[0] > 0).sum()
        dice = 2 * inter / ((enderezada[0] > 0).sum() + (etiqueta[0] > 0).sum())
        self.assertGreater(dice, 0.95)

    def test_mip_respeta_spacing(self):
        vol = np.full((4, 50, 100), -1000, dtype=np.int16)
        vol[:, 20:30, 40:60] = 800
        s = nz.mip_axial_hueso(vol, spacing_xyz=(0.5, 1.0, 1.0))
        self.assertEqual(s.shape, (50, 50))     # x a 0.5 mm → la mitad de píxeles a 1 mm


def _a_pixeles(silueta_mm, mm_x, mm_y):
    """Dibuja una silueta física (1 mm/píxel) en una grilla de píxeles no cuadrados, como el caché."""
    return (ndi.zoom(silueta_mm, (1 / mm_y, 1 / mm_x), order=0) > 0.5).astype(np.uint8)


def _dice(a, b):
    a, b = a > 0, b > 0
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


class TestPaso2EnCache(unittest.TestCase):
    """El paso 2 aplicado al caché del modelo (uint8, píxeles posiblemente no cuadrados)."""

    def test_giro_en_mm_igual_a_rotate_con_pixeles_cuadrados(self):
        v = np.zeros((2, 80, 80), dtype=np.float32)
        v[:, 20:35, 15:60] = 1
        v[:, 50:70, 30:40] = 2
        for angulo in (20, -35):
            esperado = ndi.rotate(v, -angulo, axes=(1, 2), reshape=False, order=1)
            obtenido = nz.corregir_giro_axial(v, angulo, relleno=0, mm_xy=(0.7, 0.7))
            np.testing.assert_allclose(obtenido, esperado, atol=1e-5)

    def test_umbral_u8_de_250_hu(self):
        self.assertAlmostEqual(nz.umbral_u8(250), 68.58, places=1)
        self.assertEqual(nz.umbral_u8(-1000), 0.0)
        self.assertEqual(nz.umbral_u8(2000), 255.0)

    def test_endereza_caso_con_pixeles_no_cuadrados(self):
        """Paciente girado 25° en mm, guardado en píxeles de 1.0 × 1.5 mm: se recupera y endereza."""
        mm_xy = (1.0, 1.5)
        recta = _a_pixeles(_pelvis_sintetica(0), *mm_xy)
        girada = _a_pixeles(_pelvis_sintetica(25), *mm_xy)
        imagenes = (girada * 200)[None].repeat(3, axis=0).astype(np.uint8)
        etiquetas = (girada * 11)[None].repeat(3, axis=0).astype(np.uint8)
        img2, lab2, est = nz.enderezar_caso(imagenes, etiquetas, mm_xy)
        self.assertAlmostEqual(est.angulo, 25, delta=2.0)
        self.assertTrue(est.corregir)
        self.assertEqual(img2.dtype, np.uint8)
        self.assertTrue(set(np.unique(lab2)) <= {0, 11})
        self.assertGreater(_dice(lab2[0], recta), 0.90)
        # Girar en píxeles (sin mm) deforma la pelvis y queda peor.
        en_pixeles = nz.corregir_giro_axial(etiquetas, est.angulo, es_etiqueta=True)
        self.assertGreater(_dice(lab2[0], recta), _dice(en_pixeles[0], recta))

    def test_paciente_recto_no_se_toca(self):
        recta = _a_pixeles(_pelvis_sintetica(0), 0.9, 1.2)
        imagenes = (recta * 200)[None].repeat(2, axis=0).astype(np.uint8)
        img2, lab2, est = nz.enderezar_caso(imagenes, None, (0.9, 1.2))
        self.assertFalse(est.corregir)
        self.assertIs(img2, imagenes)
        self.assertIsNone(lab2)


DATOS = RAIZ / "data" / "raw"


@unittest.skipUnless(DATOS.exists(), "Sin data/raw: se omite la comparación con los 100 casos")
class TestCasosReales(unittest.TestCase):
    def test_igual_que_to_canonical_en_los_100(self):
        import orientacion
        import pengwin_io
        from build_case_cache import set_raw_root
        set_raw_root(str(DATOS))     # otra prueba puede haber apuntado pengwin_io a una carpeta temporal
        for n in range(1, 101):
            caso = f"{n:03d}"
            img = sitk.ReadImage(str(pengwin_io.find_image_path(caso)))
            esperado = orientacion.to_canonical(sitk.GetArrayFromImage(img), caso)
            out, _, _ = nz.a_orientacion_estandar(img)
            with self.subTest(caso=caso):
                np.testing.assert_array_equal(sitk.GetArrayFromImage(out), esperado)


if __name__ == "__main__":
    unittest.main()
