import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demo_export import build_id_volume, compare_tables, fragment_meshes, fragment_table


def two_blocks(gap: int = 5, fill_between: bool = True):
    """Sacro con un bloque principal y uno secundario separados ``gap`` vóxeles en x."""
    shape = (8, 20, 60)
    semantic = np.zeros(shape, np.uint8)
    primary_secondary = np.zeros(shape, np.uint8)
    semantic[:, :, :20] = 1
    primary_secondary[:, :, :20] = 1                       # principal
    start = 20 + gap
    semantic[:, :, start:start + 5] = 1
    primary_secondary[:, :, start:start + 5] = 2           # secundario
    if fill_between:
        semantic[:, :, 20:start] = 1                       # región sin etiqueta de instancia entre ambos
    return semantic, primary_secondary


class IdVolumeAndDistances(unittest.TestCase):
    def test_gap_between_fragments_is_not_filled(self):
        semantic, ps = two_blocks(gap=5)
        volume, method = build_id_volume(semantic, ps)
        self.assertEqual(int((volume == 1).sum()), 8 * 20 * 20)       # principal intacto
        self.assertEqual(int((volume == 2).sum()), 8 * 20 * 5)        # un secundario con ID 2
        self.assertEqual(int(volume[:, :, 20:25].sum()), 0)           # el hueco sigue vacío
        self.assertTrue(method)

    def test_distance_uses_the_header_spacing(self):
        semantic, ps = two_blocks(gap=5)
        volume, _ = build_id_volume(semantic, ps)
        spacing = (0.8, 0.9, 1.2)                                    # (x, y, z)
        rows = {r["id"]: r for r in fragment_table(volume, spacing)}
        self.assertTrue(rows[1]["is_main"])
        self.assertIsNone(rows[1]["distance_mm"])
        # hueco de 5 vóxeles en x: borde a borde = 5 * 0.8 mm
        self.assertAlmostEqual(rows[2]["distance_mm"], 5 * 0.8, places=6)
        self.assertAlmostEqual(rows[2]["volume_mm3"], 8 * 20 * 5 * 0.8 * 0.9 * 1.2, places=3)

    def test_stray_bits_next_to_a_fragment_do_not_fake_contact(self):
        semantic, ps = two_blocks(gap=5)
        ps[3:5, 8:10, 22:24] = 1            # 8 vóxeles "primarios" sueltos, a un vóxel del fragmento secundario
        volume, _ = build_id_volume(semantic, ps)
        self.assertEqual(int((volume == 1).sum()), 8 * 20 * 20)       # el principal es solo el bloque grande
        rows = {r["id"]: r for r in fragment_table(volume, (0.8, 0.8, 0.8))}
        self.assertAlmostEqual(rows[2]["distance_mm"], 5 * 0.8, places=6)

    def test_touching_fragments_measure_zero(self):
        semantic, ps = two_blocks(gap=0)
        volume, _ = build_id_volume(semantic, ps)
        rows = {r["id"]: r for r in fragment_table(volume, (0.8, 0.8, 0.8))}
        self.assertEqual(rows[2]["distance_mm"], 0.0)

    def test_comparison_pairs_the_predicted_fragment_with_the_real_one(self):
        semantic, ps = two_blocks(gap=5)
        volume, _ = build_id_volume(semantic, ps)
        spacing = (0.8, 0.8, 0.8)
        truth = np.zeros_like(volume)
        truth[:, :, :20] = 1
        truth[:, :, 26:31] = 2                                       # real: 6 vóxeles de separación
        pred_rows, gt_rows = fragment_table(volume, spacing), fragment_table(truth, spacing)
        comparison = compare_tables(pred_rows, gt_rows, volume, truth)
        self.assertEqual(len(comparison), 1)
        row = comparison[0]
        self.assertEqual((row["gt_id"], row["pred_id"]), (2, 2))
        self.assertAlmostEqual(row["gt_mm"], 6 * 0.8, places=6)
        self.assertAlmostEqual(row["pred_mm"], 5 * 0.8, places=6)
        self.assertAlmostEqual(row["abs_error_mm"], 0.8, places=6)

    def test_meshes_are_in_millimetres(self):
        semantic, ps = two_blocks(gap=5)
        volume, _ = build_id_volume(semantic, ps)
        meshes = fragment_meshes(volume, (0.5, 0.5, 2.0), step=1)
        self.assertEqual(sorted(meshes), [1, 2])
        verts, faces = meshes[2]
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(faces.shape[1], 3)
        x_mm = verts[:, 2]
        # el bloque secundario ocupa x = 25..29 vóxeles → entre ~12 y ~15 mm
        self.assertGreater(float(x_mm.min()), 11.0)
        self.assertLess(float(x_mm.max()), 16.0)


def hollow_fragments():
    """Principal del coxal izquierdo con una cavidad cerrada y un secundario separado 4 vóxeles."""
    volume = np.zeros((20, 30, 50), np.uint8)
    volume[2:18, 4:26, 4:26] = 11
    volume[7:13, 10:20, 10:20] = 0                 # cavidad cerrada dentro del principal
    volume[6:14, 10:20, 30:38] = 12                # secundario, a 4 vóxeles en x
    return volume


class MeshFillIsVisualOnly(unittest.TestCase):
    def test_fill_does_not_touch_ids_fragments_or_distances(self):
        volume = hollow_fragments()
        spacing = (0.8, 0.9, 1.2)
        before = volume.copy()
        rows_before = fragment_table(volume, spacing)
        meshes = fragment_meshes(volume, spacing, step=1)
        np.testing.assert_array_equal(volume, before)                  # el volumen de IDs no cambia
        self.assertEqual(sorted(meshes), [11, 12])                      # mismos fragmentos, ninguno se une
        self.assertEqual(fragment_table(volume, spacing), rows_before)  # tabla y distancias idénticas
        self.assertAlmostEqual(rows_before[1]["distance_mm"], 4 * 0.8, places=6)

    def test_fill_closes_the_cavity_in_the_mesh(self):
        volume = hollow_fragments()
        spacing = (1.0, 1.0, 1.0)
        verts, _ = fragment_meshes(volume, spacing, step=1)[11]
        # La superficie exterior está en z ≈ 1.5/17.5 e y, x ≈ 3.5/25.5. Sin relleno habría otra
        # superficie alrededor de la cavidad (z 6.5-12.5, y y x 9.5-19.5), lejos de la exterior.
        inside = ((verts[:, 0] > 4) & (verts[:, 0] < 15) & (verts[:, 1] > 6) & (verts[:, 1] < 23)
                  & (verts[:, 2] > 6) & (verts[:, 2] < 23))
        self.assertEqual(int(inside.sum()), 0)



class EnderezadoDeLaDemo(unittest.TestCase):
    """La demo endereza solo giros claros: GIRO_MINIMO_DEMO grados o más."""

    def test_umbral_de_la_demo(self):
        from demo_export import GIRO_MINIMO_DEMO
        from normalizacion import estimar_giro_axial
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_normalizacion import _pelvis_sintetica
        self.assertFalse(estimar_giro_axial(_pelvis_sintetica(6.0), giro_minimo=GIRO_MINIMO_DEMO).corregir)
        grande = estimar_giro_axial(_pelvis_sintetica(30.0), giro_minimo=GIRO_MINIMO_DEMO)
        self.assertTrue(grande.corregir)
        self.assertLess(abs(abs(grande.angulo) - 30.0), 3.0)


if __name__ == "__main__":
    unittest.main()


class CajaDelHueso(unittest.TestCase):
    def test_recortar_al_hueso_da_la_misma_tabla(self):
        from demo_export import _caja_hueso, _desplazar
        volume = np.zeros((20, 60, 70), np.uint8)
        volume[5:12, 10:20, 15:30] = 11            # principal del coxal izquierdo
        volume[6:10, 10:20, 36:40] = 12            # fragmento a 6 vóxeles en x
        spacing = (0.8, 0.9, 1.2)
        caja = _caja_hueso(volume, None)
        self.assertEqual(caja, (slice(3, 14), slice(8, 22), slice(13, 42)))
        origen = np.array([c.start for c in caja])
        completa = fragment_table(volume, spacing)
        recortada = _desplazar(fragment_table(volume[caja], spacing), origen)
        for a, b in zip(completa, recortada):
            self.assertEqual({k: v for k, v in a.items() if k != "centroid_zyx_vox"},
                             {k: v for k, v in b.items() if k != "centroid_zyx_vox"})
            np.testing.assert_allclose(a["centroid_zyx_vox"], b["centroid_zyx_vox"])
        self.assertEqual(_caja_hueso(np.zeros((4, 5, 6), np.uint8)), (slice(0, 4), slice(0, 5), slice(0, 6)))
