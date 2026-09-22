from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from week9_detection import (CBAM, DetectionModel, boxes_from_label,
                              decode_predictions, detection_loss,
                              labels_to_grid, nms)


def logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-5, 1 - 1e-5))


class Week9DetectionTests(unittest.TestCase):
    def test_model_shape_and_cbam_position(self) -> None:
        model = DetectionModel(base_channels=8)
        self.assertIsInstance(model.backbone.cbam, CBAM)
        output = model(torch.zeros(2, 1, 256, 256))
        self.assertEqual(tuple(output.shape), (2, 16, 16, 8))

    def test_target_encode_decode_round_trip(self) -> None:
        label = torch.zeros(1, 64, 64, dtype=torch.long)
        label[0, 8:24, 4:20] = 1
        label[0, 30:55, 38:61] = 21
        target = labels_to_grid(label)
        self.assertEqual(int(target[..., 0].sum()), 2)

        # Construye logits perfectos a partir del target para probar que la
        # decodificación invierte la codificación geométrica.
        pred = torch.full_like(target, -12.0)
        positives = target[..., 0] == 1
        pred[..., 0][positives] = 12.0
        pred[..., 1:5][positives] = logit(target[..., 1:5][positives])
        for index in torch.nonzero(positives):
            b, i, j = index.tolist()
            cls = int(target[b, i, j, 5:].argmax())
            pred[b, i, j, 5:] = -12.0
            pred[b, i, j, 5 + cls] = 12.0

        decoded = decode_predictions(pred, confidence=0.9)[0]
        truth = boxes_from_label(label)[0]
        self.assertEqual(len(decoded), len(truth))
        for gt in truth:
            candidate = next(item for item in decoded if item["class_id"] == gt["class_id"])
            self.assertTrue(torch.allclose(torch.tensor(candidate["box"]),
                                           torch.tensor(gt["box"]), atol=1e-4))

    def test_nms_suppresses_overlapping_lower_score(self) -> None:
        boxes = torch.tensor([[0.1, 0.1, 0.6, 0.6],
                              [0.12, 0.12, 0.58, 0.58],
                              [0.7, 0.7, 0.9, 0.9]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        self.assertEqual(nms(boxes, scores, iou_threshold=0.5).tolist(), [0, 2])

    def test_loss_is_finite_and_backpropagates(self) -> None:
        model = DetectionModel(base_channels=4)
        image = torch.rand(1, 1, 64, 64)
        label = torch.zeros(1, 64, 64, dtype=torch.long)
        label[0, 18:46, 20:44] = 11
        pred = model(image)
        loss, metrics = detection_loss(pred, labels_to_grid(label))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["positive_cells"], 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
