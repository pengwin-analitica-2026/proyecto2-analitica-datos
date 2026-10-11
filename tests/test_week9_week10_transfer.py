import sys
import tempfile
import unittest
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from train_s10 import load_week9_initialization
from week10_multitask import PengwinMultiTaskModel
from week9_detection import DetectionModel, detection_model_from_checkpoint


class Week9ToWeek10TransferTests(unittest.TestCase):
    def _checkpoint(self, path: Path, in_channels: int = 3, base_channels: int = 4):
        torch.manual_seed(7)
        model = DetectionModel(base_channels=base_channels, in_channels=in_channels,
                               use_cbam=True, cbam_stages=1)
        checkpoint = {
            "model": model.state_dict(),
            "base_channels": base_channels,
            "in_channels": in_channels,
            "context": (in_channels - 1) // 2,
            "use_cbam": True,
            "cbam_stages": 1,
            "best_epoch": 2,
            "best_val_f1": 0.5,
            "aug_geo": 1.0,
            "aug_rot_deg": 35.0,
            "aug_mm": True,
        }
        torch.save(checkpoint, path)
        return model, checkpoint

    def test_week9_checkpoint_round_trip_keeps_three_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "week9.pth"
            original, checkpoint = self._checkpoint(path)
            restored = detection_model_from_checkpoint(checkpoint)
            self.assertEqual(restored.in_channels, 3)
            x = torch.rand(2, 3, 64, 64)
            original.eval(); restored.eval()
            with torch.inference_mode():
                self.assertTrue(torch.equal(original(x), restored(x)))

    def test_week10_loads_backbone_and_detector_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "week9.pth"
            week9, _ = self._checkpoint(path)
            week10 = PengwinMultiTaskModel(base_channels=4, in_channels=3,
                                           instance_mode="primary")
            metadata = load_week9_initialization(week10, path, torch.device("cpu"))
            inherited = {
                **{f"backbone.{k}": v for k, v in week9.backbone.state_dict().items()},
                **{f"detection_head.{k}": v for k, v in week9.detection_head.state_dict().items()},
            }
            state = week10.state_dict()
            self.assertTrue(all(torch.equal(state[key], value) for key, value in inherited.items()))
            self.assertGreater(metadata["backbone_tensors"], 0)
            self.assertGreater(metadata["detection_tensors"], 0)
            self.assertTrue(metadata["aug_mm"])
            self.assertEqual(metadata["aug_rot_deg"], 35.0)

    def test_week10_rejects_checkpoint_with_wrong_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "week9_2d.pth"
            self._checkpoint(path, in_channels=1)
            week10 = PengwinMultiTaskModel(base_channels=4, in_channels=3,
                                           instance_mode="primary")
            with self.assertRaisesRegex(ValueError, "Canales incompatibles"):
                load_week9_initialization(week10, path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
