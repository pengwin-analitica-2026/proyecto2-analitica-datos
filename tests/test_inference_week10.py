import unittest

import numpy as np
import torch

from src.inference_week10 import infer_slice, region_for_fragment


class TestInferenceWeek10(unittest.TestCase):
    def test_fragment_region_mapping(self):
        self.assertEqual(region_for_fragment(1), 0)
        self.assertEqual(region_for_fragment(11), 1)
        self.assertEqual(region_for_fragment(30), 2)
        self.assertIsNone(region_for_fragment(0))

    def test_inference_returns_masks_and_serializable_detections(self):
        class Dummy(torch.nn.Module):
            def forward(self, x):
                b, _, h, w = x.shape
                detection = torch.full((b, h // 16, w // 16, 8), -10.0)
                detection[:, 0, 0, 0] = 10.0
                detection[:, 0, 0, 5] = 10.0
                detection[:, 0, 0, 1:5] = 0.0
                semantic = torch.zeros((b, 4, h, w))
                semantic[:, 1] = 2.0
                instance = torch.zeros((b, 31, h, w))
                instance[:, 1] = 3.0
                return {"detection": detection, "semantic": semantic, "instance": instance}

        result = infer_slice(Dummy(), torch.zeros((1, 1, 64, 64)))
        self.assertEqual(result["semantic"].shape, (64, 64))
        self.assertEqual(result["fragments"].shape, (64, 64))
        self.assertEqual(result["detections"][0]["fragment_ids"], [1])


if __name__ == "__main__":
    unittest.main()
