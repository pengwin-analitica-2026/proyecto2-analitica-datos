from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from week9_detection import labels_to_grid
from week10_multitask import (PengwinMultiTaskModel, labels_to_presence,
                              labels_to_semantic, multitask_loss)


class Week10MultiTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = torch.zeros(2, 64, 64, dtype=torch.long)
        self.labels[0, 10:30, 4:22] = 1
        self.labels[0, 25:55, 35:60] = 21
        self.labels[1, 18:48, 20:44] = 11

    def test_target_mapping(self) -> None:
        semantic = labels_to_semantic(self.labels)
        self.assertEqual(sorted(torch.unique(semantic).tolist()), [0, 1, 2, 3])
        self.assertEqual(labels_to_presence(self.labels).tolist(),
                         [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    def test_three_heads_shapes(self) -> None:
        model = PengwinMultiTaskModel(base_channels=4)
        outputs = model(torch.rand(2, 1, 64, 64))
        self.assertEqual(tuple(outputs["classification"].shape), (2, 3))
        self.assertEqual(tuple(outputs["detection"].shape), (2, 4, 4, 8))
        self.assertEqual(tuple(outputs["semantic"].shape), (2, 4, 64, 64))
        self.assertEqual(tuple(outputs["instance"].shape), (2, 31, 64, 64))

    def test_composite_loss_backpropagates_to_all_heads(self) -> None:
        model = PengwinMultiTaskModel(base_channels=4)
        outputs = model(torch.rand(2, 1, 64, 64))
        loss, parts = multitask_loss(outputs, labels_to_grid(self.labels), self.labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"total", "detection", "classification", "semantic",
                                      "instance", "det_objectness", "det_box",
                                      "det_classification"})
        for head in (model.classification_head, model.detection_head, model.segmentation_head):
            self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))


if __name__ == "__main__":
    unittest.main()
