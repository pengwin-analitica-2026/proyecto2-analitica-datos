"""Controles de calidad de la entrada: espaciado imposible en la cabecera y cortes con metal."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from calidad_entrada import (aviso_metal, marcar_fragmentos_con_metal, revisar_espaciado,  # noqa: E402
                             revisar_metal)
from test_volumen_entrada import a_bytes, volumen_prueba  # noqa: E402
from volumen_entrada import leer_subida  # noqa: E402


def fila(fragment_id, z0, z1, principal=False):
    return {"id": fragment_id, "is_main": principal, "z_range": [z0, z1], "distance_mm": None if principal else 5.0}


class TestEspaciado(unittest.TestCase):
    def test_ct_normal_no_avisa(self):
        self.assertEqual(revisar_espaciado((0.78, 0.78, 0.8)), [])
        self.assertEqual(revisar_espaciado((0.94, 0.94, 5.0)), [])

    def test_cien_mm_entre_cortes_avisa(self):
        # El caso G3_P1_A del profesor: ElementSpacing = 0.938 0.938 100.
        avisos = revisar_espaciado((0.938, 0.938, 100.0))
        self.assertEqual(len(avisos), 1)
        self.assertEqual(avisos[0]["tipo"], "espaciado")
        self.assertIn("100 mm", avisos[0]["titulo"])
        self.assertIn("Grosor entre cortes", avisos[0]["texto"])

    def test_pixel_imposible_avisa(self):
        avisos = revisar_espaciado((5.0, 5.0, 1.0))
        self.assertEqual([a["tipo"] for a in avisos], ["espaciado"])
        self.assertIn("píxel", avisos[0]["titulo"])


class TestMetal(unittest.TestCase):
    def setUp(self):
        self.volumen = np.full((12, 32, 32), 40, dtype=np.int16)        # tejido blando
        self.volumen[:, 10:20, 10:20] = 600                              # hueso
        self.volumen[5:8, 2:10, 2:10] = 8000                             # un clavo en los cortes 5 a 7

    def test_encuentra_los_cortes_con_metal(self):
        resumen = revisar_metal(self.volumen, (1.0, 1.0, 20.0))
        self.assertEqual(resumen["cortes"], [5, 6, 7])
        self.assertEqual(resumen["rango_z"], [5, 7])
        self.assertAlmostEqual(resumen["volumen_mm3"], 3 * 64 * 20.0)
        self.assertTrue(resumen["avisar"])

    def test_el_hueso_no_cuenta_como_metal(self):
        sin_metal = self.volumen.copy()
        sin_metal[sin_metal > 3000] = 1800                                # cortical densa
        resumen = revisar_metal(sin_metal, (1.0, 1.0, 1.0))
        self.assertEqual(resumen["n_cortes"], 0)
        self.assertFalse(resumen["avisar"])
        self.assertIsNone(aviso_metal(resumen, 0))

    def test_poco_metal_no_avisa(self):
        resumen = revisar_metal(self.volumen, (0.5, 0.5, 0.5))           # 48 mm³: un tornillo pequeño
        self.assertFalse(resumen["avisar"])
        self.assertIsNone(aviso_metal(resumen, 0))

    def test_marca_solo_secundarios_en_cortes_con_metal(self):
        filas = [fila(11, 0, 11, principal=True), fila(12, 5, 7), fila(13, 6, 9), fila(14, 0, 3)]
        marcados = marcar_fragmentos_con_metal(filas, [5, 6, 7])
        self.assertEqual(marcados, 2)
        self.assertEqual([f["metal"] for f in filas], [False, True, True, False])   # 13: 2 de 4 cortes

    def test_sin_aviso_no_se_marca_nada(self):
        filas = [fila(12, 5, 7)]
        self.assertEqual(marcar_fragmentos_con_metal(filas, []), 0)
        self.assertFalse(filas[0]["metal"])

    def test_metal_de_exactamente_3000_hu_cuenta(self):
        # G3_P1_A: cilindros sintéticos con todos sus vóxeles en 3000 HU exactos.
        volumen = np.full((4, 32, 32), 40, dtype=np.int16)
        volumen[1:3, 5:15, 5:15] = 3000
        self.assertEqual(revisar_metal(volumen, (1.0, 1.0, 10.0))["cortes"], [1, 2])

    def test_texto_del_aviso(self):
        resumen = revisar_metal(self.volumen, (1.0, 1.0, 20.0))
        aviso = aviso_metal(resumen, 2)
        self.assertEqual(aviso["tipo"], "metal")
        self.assertIn("3.840 mm³", aviso["texto"])
        self.assertIn("3 cortes, entre el 5 y el 7", aviso["texto"])
        self.assertIn("2 fragmentos", aviso["texto"])


class TestDescribirEntrada(unittest.TestCase):
    def test_describe_lo_que_trae_el_archivo(self):
        datos = a_bytes(volumen_prueba(), ".mha", spacing=(0.8, 0.9, 1.5))
        e = leer_subida([("caso.mha", datos)]).entrada
        self.assertEqual(e["tamano_xyz"], [24, 20, 12])
        self.assertEqual(e["espaciado_xyz"], [0.8, 0.9, 1.5])
        self.assertEqual(e["direccion"], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(e["n_archivos"], 1)
        self.assertLessEqual(e["hu"]["min"], e["hu"]["p50"])
        self.assertIn("16-bit", e["tipo_voxel"])
        self.assertEqual(e["cambios"], [])


class TestCorregirGrosor(unittest.TestCase):
    def test_grosor_reemplaza_el_de_la_cabecera_en_un_mha(self):
        datos = a_bytes(volumen_prueba(), ".mha", spacing=(0.938, 0.938, 100.0))
        sin_corregir = leer_subida([("G3.mha", datos)])
        self.assertAlmostEqual(sin_corregir.spacing_xyz[2], 100.0)
        corregido = leer_subida([("G3.mha", datos)], grosor_mm=1.6)
        self.assertAlmostEqual(corregido.spacing_xyz[2], 1.6)
        self.assertAlmostEqual(corregido.spacing_xyz[0], 0.938, places=3)
        self.assertIn("corregido a 1.6 mm", corregido.fuente)
        self.assertIn("decía 100 mm", corregido.fuente)
        self.assertEqual(corregido.entrada["espaciado_xyz"][2], 100.0)            # lo que traía el archivo
        self.assertIn("100 → 1.6 mm", corregido.entrada["cambios"][0])
        np.testing.assert_array_equal(corregido.image, sin_corregir.image)   # los HU no cambian


if __name__ == "__main__":
    unittest.main()
