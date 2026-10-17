"""La cuantización debe correr de punta a punta y calibrar solo con casos de entrenamiento."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_evaluate_volumes import synthetic_case
from cuantizar import calibration_slices, quantize_int8
from week10_multitask import PengwinMultiTaskModel


def make_cache(root: Path, size: int = 64) -> Path:
    """Dos casos de entrenamiento y uno de validación, con el índice que pide el post-proceso 3D."""
    cache = root / f"cache_{size}"
    cache.mkdir(parents=True)
    rows = []
    for number, split in enumerate(("train", "train", "val"), 1):
        images, labels = synthetic_case(depth=12, size=size, seed=number)
        # Mismos campos que build_case_cache: train_s10 los usa para el aumento en milímetros.
        np.savez(cache / f"{number:03d}.npz", images=images, labels=labels,
                 spacing_xyz=np.array([1.0, 1.0, 2.0]), orig_shape=np.asarray(images.shape),
                 size=np.array(size), mode=np.array("stretch"))
        rows.append({"case_id": f"{number:03d}", "split": split, "shape_x": size, "shape_y": size,
                     "spacing_x_mm": 1.0, "spacing_y_mm": 1.0, "spacing_z_mm": 2.0})
    with open(root / "index.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (root / "splits.json").write_text(json.dumps({"train": [1, 2], "val": [3], "test": []}))
    return cache


class CuantizarTests(unittest.TestCase):
    def test_calibration_uses_only_training_cases(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cache = make_cache(root)
            (cache / "001.npz").unlink(); (cache / "002.npz").unlink()   # solo queda el de validación
            with self.assertRaises(SystemExit):
                calibration_slices(cache, root / "splits.json", 8, 4, canonical=False)

    def test_int8_is_smaller_and_close_to_fp32(self):
        torch.manual_seed(0)
        model = PengwinMultiTaskModel(base_channels=8, instance_mode="primary").eval()
        x = torch.rand(8, 1, 64, 64)
        quantized = quantize_int8(model, x, keep_float=False)
        with torch.inference_mode():
            a, b = model(x[:2]), quantized(x[:2])
        self.assertEqual(set(a), set(b))
        for key in ("semantic", "instance", "classification", "detection"):
            self.assertEqual(tuple(a[key].shape), tuple(b[key].shape))
        self.assertEqual(quantized.instance_mode, "primary")

    def test_end_to_end_reports_every_variant(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_cache(root)
            checkpoint = root / "m.pth"
            subprocess.run(
                [sys.executable, str(SRC / "train_s10.py"), "--cache-root", str(root), "--size", "64",
                 "--epochs", "1", "--batch-size", "4", "--base-channels", "4", "--margin", "1",
                 "--val-stride", "2", "--per-epoch", "8", "--output", str(checkpoint)],
                check=True, capture_output=True)
            out = root / "q.json"
            run = subprocess.run(
                [sys.executable, str(SRC / "cuantizar.py"), "--checkpoint", str(checkpoint),
                 "--cache-dir", str(root / "cache_64"), "--splits", str(root / "splits.json"),
                 "--split", "val", "--tta", "--repeats", "3", "--calib-per-case", "4",
                 "--variants", "fp32", "bf16", "int8", "int8_cbam_fp32", "--output", str(out)],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            result = json.loads(out.read_text())["variantes"]
            self.assertEqual(set(result), {"fp32", "bf16", "int8", "int8_cbam_fp32"})
            self.assertLess(result["int8"]["tamano_mb"], result["fp32"]["tamano_mb"])
            for variant in ("bf16", "int8", "int8_cbam_fp32"):
                self.assertIn("delta_dice_fragmento", result[variant])


if __name__ == "__main__":
    unittest.main()
