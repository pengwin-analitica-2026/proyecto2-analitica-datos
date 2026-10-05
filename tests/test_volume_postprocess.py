import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volume_postprocess import (cache_spacing_xyz, fragment_distance_table, match_fragments,
                                postprocess_3d, surface_distances)


def synthetic_volume():
    """Coxal izquierdo: principal grande y un fragmento secundario a 6 vóxeles de distancia."""
    labels = np.zeros((16, 40, 40), dtype=np.uint8)
    labels[2:14, 5:30, 5:20] = 11
    labels[5:9, 5:12, 26:32] = 12          # fragmento a 6 vóxeles en x del principal
    return labels


class PostprocessTests(unittest.TestCase):
    def test_perfect_prediction_is_recovered(self):
        truth = synthetic_volume()
        semantic = np.where((truth >= 11) & (truth <= 20), 2, 0).astype(np.uint8)
        ps = np.zeros_like(truth); ps[truth == 11] = 3; ps[truth == 12] = 4
        out = postprocess_3d(semantic, ps, min_voxels=5, min_region_voxels=50)
        self.assertTrue(np.array_equal(out, truth))

    def test_3d_removes_flicker_and_fixes_primary(self):
        truth = synthetic_volume()
        semantic = np.where((truth >= 11) & (truth <= 20), 2, 0).astype(np.uint8)
        ps = np.zeros_like(truth); ps[truth == 11] = 3; ps[truth == 12] = 4
        ps[7, 20:24, 8:10] = 3                      # un "primario" aislado dentro del fragmento? no: en el hueso
        ps[3, 10:14, 10:14] = 4                     # parpadeo secundario de 1 corte dentro del principal
        out = postprocess_3d(semantic, ps, min_voxels=100, min_region_voxels=50)
        self.assertEqual(int((out == 12).sum()), int((truth == 12).sum()))   # el parpadeo se absorbe
        self.assertEqual(int((out == 11).sum()), int((truth == 11).sum()))

    def test_noise_region_is_dropped(self):
        truth = synthetic_volume()
        semantic = np.where((truth >= 11) & (truth <= 20), 2, 0).astype(np.uint8)
        semantic[0, 0, 0:3] = 1                     # sacro "fantasma" de 3 vóxeles
        out = postprocess_3d(semantic, np.zeros_like(truth), min_voxels=5, min_region_voxels=50)
        self.assertEqual(int((out == 1).sum()), 0)

    def test_distance_matches_geometry_and_error_is_zero_for_perfect(self):
        truth = synthetic_volume()
        rows = fragment_distance_table(truth, truth, (1.0, 1.0, 1.0))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["ground_truth_mm"], 6.0, places=4)
        self.assertAlmostEqual(rows[0]["absolute_error_mm"], 0.0, places=6)
        scaled = fragment_distance_table(truth, truth, (0.5, 0.5, 2.0))
        self.assertAlmostEqual(scaled[0]["ground_truth_mm"], 3.0, places=4)

    def test_undetected_fragment_has_no_prediction(self):
        truth = synthetic_volume()
        pred = truth.copy(); pred[pred == 12] = 0
        row = fragment_distance_table(pred, truth, (1.0, 1.0, 1.0))[0]
        self.assertIsNone(row["predicted_mm"]); self.assertEqual(row["dice"], 0.0)

    def test_surface_distances_of_shifted_cube(self):
        a = np.zeros((20, 20, 20), bool); a[5:10, 5:10, 5:10] = True
        b = np.zeros_like(a); b[5:10, 5:10, 8:13] = True
        assd, hd95 = surface_distances(a, a, (1, 1, 1))
        self.assertEqual((assd, hd95), (0.0, 0.0))
        assd, hd95 = surface_distances(a, b, (1, 1, 1))
        self.assertGreater(assd, 0); self.assertLessEqual(assd, hd95 + 1e-6)

    def test_match_fragments_uses_each_prediction_once(self):
        truth = synthetic_volume()
        pred = truth.copy(); pred[pred == 12] = 11
        rows = {m["gt_id"]: m for m in match_fragments(pred, truth)}
        self.assertEqual(rows[12]["dice"], 0.0)

    def test_cache_spacing_scales_inplane_axes(self):
        row = {"spacing_x_mm": "0.8", "spacing_y_mm": "0.9", "spacing_z_mm": "1.5",
               "shape_x": "512", "shape_y": "256"}
        sx, sy, sz = cache_spacing_xyz(row, 384)
        self.assertAlmostEqual(sx, 0.8 * 512 / 384); self.assertAlmostEqual(sy, 0.9 * 256 / 384)
        self.assertEqual(sz, 1.5)


if __name__ == "__main__":
    unittest.main()
