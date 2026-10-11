import csv, sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from train_s10 import epoch_indices, load_slices, to_batch
from week10_multitask import PengwinMultiTaskModel


class TrainS10Tests(unittest.TestCase):
    def test_epoch_indices_include_all_bone_and_some_negatives(self):
        has_bone = np.array([False] * 5 + [True] * 10 + [False] * 5)
        idx = epoch_indices(has_bone, np.random.default_rng(0), 0.3, None)
        self.assertEqual(int(has_bone[idx].sum()), 10)
        self.assertEqual(int((~has_bone[idx]).sum()), 3)
        self.assertEqual(len(set(idx.tolist())), len(idx))

    def test_cbam_options_change_parameters(self):
        n = lambda **k: sum(p.numel() for p in PengwinMultiTaskModel(base_channels=4, **k).parameters())
        self.assertLess(n(use_cbam=False), n(use_cbam=True))
        self.assertLess(n(cbam_stages=1), n(cbam_stages=3))
        out = PengwinMultiTaskModel(base_channels=4, cbam_stages=3, learnable_gamma=True)(torch.zeros(1, 1, 64, 64))
        self.assertIn("semantic", out)

    def test_load_slices_margin(self):
        with tempfile.TemporaryDirectory() as d:
            labels = np.zeros((20, 8, 8), np.uint8); labels[8:12, 2:4, 2:4] = 1
            np.savez(Path(d) / "001.npz", images=np.zeros((20, 8, 8), np.uint8), labels=labels)
            out = load_slices(Path(d), ["001"], margin=2)
            self.assertEqual(len(out["images"]), 8)   # 4 con hueso + 2 de margen por lado
            self.assertEqual(int(out["has_bone"].sum()), 4)
            x, y = to_batch(out["images"], out["labels"])
            self.assertEqual(tuple(x.shape), (8, 1, 8, 8)); self.assertEqual(y.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
