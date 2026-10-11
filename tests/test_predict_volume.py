import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_case_cache import to_uint8
from pengwin_io import resize_slice, window_hu
from predict_volume import cache_spacing_xyz, predict_slice, predict_volume, preprocess_volume
from week10_multitask import PengwinMultiTaskModel


class PredictVolume(unittest.TestCase):
    def test_preprocess_matches_the_training_cache_recipe(self):
        rng = np.random.default_rng(0)
        volume = rng.integers(-1000, 1500, size=(3, 40, 50)).astype(np.int16)
        out = preprocess_volume(volume, 32)
        self.assertEqual((out.dtype, out.shape), (np.uint8, (3, 32, 32)))
        for z in range(3):
            np.testing.assert_array_equal(out[z], to_uint8(resize_slice(window_hu(volume[z]), 32)))

    def test_cache_spacing_stretches_each_axis(self):
        # corte original 100 (y) × 200 (x) llevado a 50×50: cada píxel cubre 2× en y y 4× en x
        sx, sy, sz = cache_spacing_xyz((0.5, 0.5, 1.25), (7, 100, 200), 50)
        self.assertEqual((sx, sy, sz), (2.0, 1.0, 1.25))

    def test_predict_volume_shapes_and_slice_latency(self):
        torch.manual_seed(0)
        model = PengwinMultiTaskModel(base_channels=4, instance_mode="primary").eval()
        images = np.random.default_rng(1).integers(0, 255, size=(5, 64, 64)).astype(np.uint8)
        out = predict_volume(model, images, torch.device("cpu"), batch_size=2, confidence=0.0)
        self.assertEqual(out["semantic"].shape, (5, 64, 64))
        self.assertEqual(out["primary_secondary"].shape, (5, 64, 64))
        self.assertEqual(out["classification"].shape, (5, 3))
        self.assertEqual((len(out["boxes_raw"]), len(out["boxes_refined"])), (5, 5))
        self.assertTrue(((out["classification"] >= 0) & (out["classification"] <= 1)).all())
        for box in out["boxes_raw"][0]:
            self.assertEqual(len(box["box"]), 4)
            self.assertIn(box["region"], ("sacro", "coxal_izquierdo", "coxal_derecho"))
        single = predict_slice(model, torch.device("cpu"), images[0])
        self.assertEqual(single["semantic"].shape, (1, 64, 64))
        self.assertGreater(single["latency_ms"], 0.0)

    def test_rejects_models_without_the_primary_secondary_map(self):
        model = PengwinMultiTaskModel(base_channels=4, instance_mode="ids").eval()
        with self.assertRaises(ValueError):
            predict_volume(model, np.zeros((1, 64, 64), np.uint8), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
