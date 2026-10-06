"""El evaluador de volumen completo debe coincidir con ``evaluate()`` sobre los mismos cortes."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate_volumes as ev
from build_case_cache import from_uint8
from pengwin_metrics import matched_instance_metrics
from train_week10_multitask import evaluate as legacy_evaluate
from week10_multitask import PengwinMultiTaskModel


def synthetic_case(depth: int = 10, size: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(depth, size, size), dtype=np.uint8)
    labels = np.zeros((depth, size, size), dtype=np.uint8)
    for z in range(2, depth - 2):            # los dos cortes de cada extremo no tienen hueso
        labels[z, 20:44, 24:40] = 1           # sacro
        labels[z, 12:50, 4:22] = 11           # coxal izquierdo, principal
        labels[z, 30:38, 8:14] = 12           # fragmento secundario
        labels[z, 12:50, 42:60] = 21          # coxal derecho
        if z % 2:
            labels[z, 14:20, 50:58] = 22
    return images, labels


class EvaluateVolumesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.images, cls.labels = synthetic_case()
        cls.models = {mode: PengwinMultiTaskModel(base_channels=4, instance_mode=mode).eval()
                      for mode in ("primary", "ids")}

    def test_matches_legacy_evaluate_on_the_same_slices(self):
        for mode, model in self.models.items():
            with self.subTest(instance_mode=mode):
                records, _ = ev.evaluate_case(model, "001", self.images, self.labels,
                                              torch.device("cpu"), batch_size=4, confidence=0.10)
                mine = ev.summarize(records, "todo")
                legacy = legacy_evaluate(model, torch.from_numpy(from_uint8(self.images))[:, None],
                                         torch.from_numpy(self.labels.astype(np.int64)),
                                         torch.device("cpu"), 4, 0.10)
                for key in ("classification_f1_macro", "classification_auc_macro",
                            "detection_iou_mean", "detection_map50", "detection_map50_95",
                            "segmentation_dice_macro", "segmentation_iou_macro",
                            "instance_dice_matched", "instance_iou_matched"):
                    self.assertAlmostEqual(mine[key], legacy[key], places=6, msg=key)

    def test_matched_fragments_equals_matched_instance_metrics(self):
        rng = np.random.default_rng(1)
        truth = self.labels[3:7]
        prediction = truth.copy()
        prediction[rng.random(prediction.shape) < 0.2] = 0
        prediction[1][prediction[1] == 12] = 13      # mismo fragmento con otro ID
        prediction[2][prediction[2] == 21] = 0        # fragmento perdido
        rows = [row for p, t in zip(prediction, truth) for row in ev.matched_fragments(p, t)]
        reference = matched_instance_metrics(prediction, truth)
        self.assertAlmostEqual(float(np.mean([r[1] for r in rows])), reference["instance_dice_matched"])
        self.assertAlmostEqual(float(np.mean([r[2] for r in rows])), reference["instance_iou_matched"])
        self.assertIn((21, 0.0, 0.0, int((truth[2] == 21).sum())), rows)

    def test_subsets_partition_the_volume(self):
        records, _ = ev.evaluate_case(self.models["primary"], "001", self.images, self.labels,
                                      torch.device("cpu"), batch_size=4)
        counts = {name: ev.summarize(records, name)["n_slices"] for name in ev.SUBSETS}
        self.assertEqual(counts["todo"], 10)
        self.assertEqual(counts["con_hueso"], 6)
        self.assertEqual(counts["sin_hueso"], 4)
        self.assertEqual(counts["centro_15_85"] + counts["extremos"], counts["con_hueso"])
        positions = [r.rel_pos for r in records if r.has_bone]
        self.assertAlmostEqual(min(positions), 0.0)
        self.assertAlmostEqual(max(positions), 1.0)

    def test_raw_detection_is_reported_separately_from_refined(self):
        records, _ = ev.evaluate_case(self.models["primary"], "001", self.images, self.labels,
                                      torch.device("cpu"), batch_size=4, confidence=0.0)
        summary = ev.summarize(records, "todo")
        for key in ("detection_map50", "detection_raw_map50", "detection_raw_one_map50",
                    "detection_raw_iou_mean", "instance_dice_primary", "instance_dice_secondary"):
            self.assertIn(key, summary)
        self.assertTrue(all(len(r.det_raw_one) <= 3 for r in records))
        self.assertGreaterEqual(summary["n_raw_boxes"], sum(len(r.det_raw_one) for r in records))

    def test_predictions_can_be_kept_for_3d_postprocessing(self):
        _, predictions = ev.evaluate_case(self.models["primary"], "001", self.images, self.labels,
                                          torch.device("cpu"), batch_size=4, keep_predictions=True)
        self.assertEqual(predictions["semantic"].shape, self.labels.shape)
        self.assertEqual(predictions["instance"].shape, self.labels.shape)
        self.assertEqual(predictions["instance"].dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
