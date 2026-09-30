from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fragment_distances import (compare_fragment_distances, edge_distance_mm,
                                measure_fragment_distances)


class FragmentDistanceTests(unittest.TestCase):
    def test_adjacent_voxels_have_zero_edge_gap(self) -> None:
        main = np.zeros((3, 3, 6), dtype=bool)
        fragment = np.zeros_like(main)
        main[1, 1, 2] = True
        fragment[1, 1, 3] = True
        self.assertEqual(edge_distance_mm(fragment, main, (0.8, 0.8, 1.2)), 0.0)

    def test_anisotropic_axis_gap_is_in_millimetres(self) -> None:
        main = np.zeros((3, 3, 8), dtype=bool)
        fragment = np.zeros_like(main)
        main[1, 1, 1] = True
        fragment[1, 1, 5] = True
        # Centros separados 4 vóxeles en x; entre bordes quedan 3 vóxeles.
        self.assertAlmostEqual(edge_distance_mm(fragment, main, (0.7, 0.8, 1.2)), 2.1)

    def test_measure_and_compare_by_fragment_id(self) -> None:
        gt = np.zeros((4, 4, 10), dtype=np.uint8)
        gt[1, 1, 1] = 1
        gt[1, 1, 5] = 2
        gt[2, 2, 2] = 11
        gt[2, 2, 6] = 12
        pred = gt.copy()
        pred[1, 1, 5] = 0
        pred[1, 1, 4] = 2
        measured = measure_fragment_distances(gt, (1.0, 1.0, 1.0))
        self.assertEqual([item.fragment_label for item in measured], [2, 12])
        rows = compare_fragment_distances(pred, gt, (1.0, 1.0, 1.0))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["absolute_error_mm"], 1.0)


if __name__ == "__main__":
    unittest.main()
