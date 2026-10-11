"""Las opciones de exploración de la Semana 10 (contexto 2.5D, focal, Tversky, supervisión
profunda, EMA, sobremuestreo de secundarios) deben entrenar, guardar y evaluarse de punta a punta."""

import csv
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
from train_s10 import epoch_indices, gather_context, load_slices
from week10_multitask import (MultiTaskLossWeights, PengwinMultiTaskModel, instance_dice_loss,
                              multitask_loss)
from week9_detection import labels_to_grid


def make_cache(root: Path, size: int = 64) -> Path:
    cache = root / f"cache_{size}"
    cache.mkdir(parents=True)
    rows = []
    for number, split in enumerate(("train", "train", "val"), 1):
        images, labels = synthetic_case(depth=12, size=size, seed=number)
        np.savez(cache / f"{number:03d}.npz", images=images, labels=labels)
        rows.append({"case_id": f"{number:03d}", "split": split})
    with open(root / "index.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "split"])
        writer.writeheader(); writer.writerows(rows)
    return cache


class ContextTests(unittest.TestCase):
    def test_gather_context_stays_inside_each_case(self):
        with tempfile.TemporaryDirectory() as d:
            cache = make_cache(Path(d))
            data = load_slices(cache, ["001", "002"], margin=1)
            first, last_of_first = 0, int(data["case_hi"][0])
            window = gather_context(data, np.array([first, last_of_first, last_of_first + 1]), 2)
            self.assertEqual(window.shape[:2], (3, 5))
            # el primer corte repite su borde, el último del caso 001 no toma cortes del 002
            self.assertTrue(np.array_equal(window[0, 0], data["images"][0]))
            self.assertTrue(np.array_equal(window[1, 4], data["images"][last_of_first]))
            self.assertTrue(np.array_equal(window[2, 0], data["images"][last_of_first + 1]))

    def test_oversample_prefers_slices_with_secondary_fragments(self):
        has_bone = np.ones(400, dtype=bool)
        secondary = np.zeros(400, dtype=bool); secondary[:40] = True
        rng = np.random.default_rng(0)
        base = sum(int(secondary[epoch_indices(has_bone, rng, 0.0, 40, secondary, 1.0)].sum())
                   for _ in range(30))
        boosted = sum(int(secondary[epoch_indices(has_bone, rng, 0.0, 40, secondary, 8.0)].sum())
                      for _ in range(30))
        self.assertGreater(boosted, 2 * base)


class LossOptionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        _, labels = synthetic_case(depth=8, size=64)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def test_every_option_gives_finite_loss_and_gradients(self):
        model = PengwinMultiTaskModel(base_channels=4, instance_mode="primary", in_channels=3,
                                      deep_supervision=True).train()
        x = torch.rand(8, 3, 64, 64)
        for name, weights in {
            "focal": MultiTaskLossWeights(focal_gamma=2.0),
            "tversky": MultiTaskLossWeights(tversky_beta=0.7),
            "deep": MultiTaskLossWeights(deep_supervision=0.3),
            "todo": MultiTaskLossWeights(focal_gamma=2.0, tversky_beta=0.7, deep_supervision=0.3,
                                         secondary_weight=3.0),
        }.items():
            with self.subTest(name):
                model.zero_grad()
                loss, parts = multitask_loss(model(x), labels_to_grid(self.labels), self.labels,
                                             weights=weights)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
        self.assertGreater(parts["deep_supervision"], 0)

    def test_options_off_reproduce_the_original_loss(self):
        model = PengwinMultiTaskModel(base_channels=4, instance_mode="primary").eval()
        out = model(torch.rand(8, 1, 64, 64))
        a, _ = multitask_loss(out, labels_to_grid(self.labels), self.labels, weights=MultiTaskLossWeights())
        b, _ = multitask_loss(out, labels_to_grid(self.labels), self.labels,
                              weights=MultiTaskLossWeights(focal_gamma=0.0, tversky_beta=0.0,
                                                           deep_supervision=0.0))
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_tversky_punishes_missed_fragment_more_than_extra_pixels(self):
        labels = torch.zeros(1, 8, 8, dtype=torch.long); labels[0, 2:4, 2:4] = 1
        miss = torch.zeros(1, 7, 8, 8); miss[:, 0] = 5          # no predice el fragmento
        extra = torch.zeros(1, 7, 8, 8); extra[:, 1] = 5        # lo cubre todo
        self.assertGreater(float(instance_dice_loss(miss, labels, 0.8)),
                           float(instance_dice_loss(extra, labels, 0.8)))


class EndToEndTests(unittest.TestCase):
    def test_train_and_evaluate_with_all_options(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_cache(root)
            out = root / "m.pth"
            subprocess.run(
                [sys.executable, str(SRC / "train_s10.py"), "--cache-root", str(root), "--size", "64",
                 "--epochs", "2", "--batch-size", "4", "--base-channels", "4", "--margin", "1",
                 "--val-stride", "2", "--eval-every", "1", "--context", "1", "--focal-gamma", "2",
                 "--tversky-beta", "0.7", "--deep-supervision-weight", "0.3", "--ema", "0.9",
                 "--secondary-oversample", "3", "--per-epoch", "10", "--augment", "--cosine",
                 "--output", str(out)], check=True, capture_output=True)
            checkpoint = torch.load(out, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["in_channels"], 3)
            self.assertTrue(checkpoint["deep_supervision"])
            result = root / "val.json"
            subprocess.run(
                [sys.executable, str(SRC / "evaluate_volumes.py"), "--checkpoint", str(out),
                 "--cache-dir", str(root / "cache_64"), "--split", "val", "--batch-size", "4",
                 "--output", str(result)], check=True, capture_output=True)
            self.assertTrue(result.exists())


if __name__ == "__main__":
    unittest.main()
