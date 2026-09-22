from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pengwin_metrics import refine_boxes_with_semantic


class RefineDetectionTests(unittest.TestCase):
    def test_semantic_extent_refines_and_deduplicates_class(self) -> None:
        semantic = np.zeros((1, 10, 10), dtype=np.uint8)
        semantic[0, 2:7, 3:9] = 1
        predictions = [[
            {"class_id": 0, "score": 0.9, "box": [0, 0, .5, .5]},
            {"class_id": 0, "score": 0.4, "box": [.2, .2, .8, .8]},
        ]]
        output = refine_boxes_with_semantic(predictions, semantic)
        self.assertEqual(len(output[0]), 1)
        self.assertEqual(output[0][0]["box"], [.3, .2, .9, .7])


if __name__ == "__main__":
    unittest.main()
