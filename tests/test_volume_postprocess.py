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


def test_sin_relleno_conserva_la_separacion_entre_fragmentos():
    """Con fill_unlabeled=True el hueco entre fragmentos se vuelve principal (distancia 0)."""
    from volume_postprocess import fragment_distance_table
    shape = (6, 16, 40)
    semantic = np.zeros(shape, dtype=np.uint8)
    semantic[:, 2:14, 2:38] = 1                       # región sacro continua (cubre el hueco)
    ps = np.zeros(shape, dtype=np.uint8)
    ps[:, 2:14, 2:14] = 1                              # principal
    ps[:, 2:14, 26:38] = 2                             # secundario separado por 12 vóxeles
    truth = np.zeros(shape, dtype=np.uint8)
    truth[:, 2:14, 2:14] = 1
    truth[:, 2:14, 26:38] = 2
    spacing = (1.0, 1.0, 1.0)
    filled = postprocess_3d(semantic, ps, min_voxels=10, min_region_voxels=10)
    raw = postprocess_3d(semantic, ps, min_voxels=10, min_region_voxels=10, fill_unlabeled=False)
    d_filled = fragment_distance_table(filled, truth, spacing)[0]["predicted_mm"]
    d_raw = fragment_distance_table(raw, truth, spacing)[0]["predicted_mm"]
    assert d_filled == 0.0
    assert d_raw > 5.0


class UmbralesEnMilimetros(unittest.TestCase):
    def test_vóxel_de_referencia_da_los_umbrales_de_siempre(self):
        from volume_postprocess import VOXEL_REFERENCIA_MM3, umbrales_en_voxeles
        lado = VOXEL_REFERENCIA_MM3 ** (1 / 3)
        self.assertEqual(umbrales_en_voxeles((lado, lado, lado)), (30, 200))

    def test_cortes_tres_veces_mas_gruesos_piden_un_tercio_de_voxeles(self):
        from volume_postprocess import VOXEL_REFERENCIA_MM3, umbrales_en_voxeles
        lado = VOXEL_REFERENCIA_MM3 ** (1 / 3)
        self.assertEqual(umbrales_en_voxeles((lado, lado, 3 * lado)), (10, 67))

    def test_fragmento_real_no_se_borra_con_cortes_gruesos(self):
        """Un fragmento desplazado de 72 vóxeles (216 mm³ con cortes de 3 mm) sobrevive en mm³.

        Al estar separado del principal forma su propio componente 3D de la región, así que en vóxeles
        lo borra el umbral de región (72 < 200); en mm³ ese umbral baja a 58 vóxeles.
        """
        from volume_postprocess import umbrales_en_voxeles
        shape = (10, 30, 40)
        semantic = np.zeros(shape, dtype=np.uint8)
        ps = np.zeros(shape, dtype=np.uint8)
        semantic[:, 2:28, 2:20] = 2
        ps[:, 2:28, 2:20] = 3                       # principal del coxal izquierdo
        semantic[4:7, 10:14, 30:36] = 2
        ps[4:7, 10:14, 30:36] = 4                   # secundario: 3 x 4 x 6 = 72 vóxeles
        spacing = (1.0, 1.0, 3.0)                   # vóxel de 3 mm³
        en_voxeles = postprocess_3d(semantic, ps, min_voxels=30)
        en_mm = postprocess_3d(semantic, ps, *umbrales_en_voxeles(spacing))
        self.assertEqual(int((en_voxeles == 12).sum()), 0)
        self.assertEqual(int((en_mm == 12).sum()), 72)
