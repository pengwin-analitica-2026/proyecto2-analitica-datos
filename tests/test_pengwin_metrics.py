from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pengwin_metrics import detection_metrics, multilabel_f1_auc, segmentation_metrics


class PengwinMetricTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self) -> None:
        logits = np.array([[4, -4, 4], [-4, 4, -4]], dtype=float)
        targets = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
        classification = multilabel_f1_auc(logits, targets)
        self.assertEqual(classification["classification_f1_macro"], 1.0)
        prediction = np.array([[[0, 1], [2, 2]]])
        segmentation = segmentation_metrics(prediction, prediction)
        self.assertEqual(segmentation["segmentation_dice_macro"], 1.0)
        detections = [[{"class_id": 0, "score": 0.9, "box": [0, 0, 1, 1]}]]
        truth = [[{"class_id": 0, "box": [0, 0, 1, 1]}]]
        detection = detection_metrics(detections, truth)
        self.assertAlmostEqual(detection["detection_iou_mean"], 1.0, places=6)
        self.assertEqual(detection["detection_map50"], 1.0)


if __name__ == "__main__":
    unittest.main()
