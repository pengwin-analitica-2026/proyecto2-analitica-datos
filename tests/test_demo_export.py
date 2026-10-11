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


if __name__ == "__main__":
    unittest.main()
