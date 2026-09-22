from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from instance_postprocess import semantic_to_instances
from pengwin_metrics import matched_instance_metrics


class InstancePostprocessTests(unittest.TestCase):
    def test_components_are_numbered_with_main_first(self) -> None:
        semantic = np.zeros((8, 8), dtype=np.uint8)
        semantic[1:5, 1:5] = 2
        semantic[6:8, 6:8] = 2
        result = semantic_to_instances(semantic, min_size=1)
        self.assertEqual(int(result[1, 1]), 11)
        self.assertEqual(int(result[6, 6]), 12)

    def test_matching_is_invariant_to_instance_ids(self) -> None:
        truth = np.zeros((1, 6, 6), dtype=np.uint8)
        truth[0, :3, :3] = 11
        prediction = np.zeros_like(truth)
        prediction[0, :3, :3] = 17
        metric = matched_instance_metrics(prediction, truth)
        self.assertEqual(metric["instance_dice_matched"], 1.0)
        self.assertEqual(metric["instance_iou_matched"], 1.0)
