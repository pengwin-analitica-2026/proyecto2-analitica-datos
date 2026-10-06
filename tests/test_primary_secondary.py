"""Pruebas de la formulación primario/secundario y del aumento sagital."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from instance_postprocess import primary_secondary_to_instances
from pengwin_metrics import matched_instance_metrics
from week9_detection import labels_to_grid
from week10_multitask import (INSTANCE_MODES, MultiTaskLossWeights, PengwinMultiTaskModel,
                              flip_sagittal, fracture_distance_weight,
                              labels_to_primary_secondary, multitask_loss)


def _synthetic_labels() -> torch.Tensor:
    labels = torch.zeros((2, 64, 64), dtype=torch.long)
    labels[:, 10:30, 10:30] = 1      # sacro primario
    labels[:, 30:36, 10:30] = 2      # sacro secundario, en contacto con el primario
    labels[:, 40:60, 5:25] = 11      # coxal izq primario
    labels[:, 40:60, 27:32] = 12     # coxal izq secundario (separado)
    labels[:, 40:60, 40:60] = 21     # coxal der primario
    return labels


class PrimarySecondaryTests(unittest.TestCase):
    def test_target_has_seven_classes_and_keeps_regions(self) -> None:
        target = labels_to_primary_secondary(_synthetic_labels())
        self.assertEqual(sorted(torch.unique(target).tolist()), [0, 1, 2, 3, 4, 5])
        self.assertTrue((target[:, 10:30, 10:30] == 1).all())
        self.assertTrue((target[:, 30:36, 10:30] == 2).all())
        self.assertTrue((target[:, 40:60, 27:32] == 4).all())

    def test_round_trip_recovers_instances_perfectly(self) -> None:
        labels = _synthetic_labels()
        recovered = primary_secondary_to_instances(labels_to_primary_secondary(labels).numpy())
        metric = matched_instance_metrics(recovered, labels.numpy())
        self.assertEqual(metric["instance_dice_matched"], 1.0)
        self.assertTrue((recovered == labels.numpy()).all())

    def test_missing_primary_promotes_largest_secondary(self) -> None:
        ps = np.zeros((32, 32), dtype=np.uint8)
        ps[2:10, 2:10] = 2    # sacro secundario grande, sin primario
        ps[20:23, 20:23] = 2  # sacro secundario pequeño
        out = primary_secondary_to_instances(ps, min_size=1)
        self.assertEqual(out[5, 5], 1)
        self.assertEqual(out[21, 21], 2)

    def test_model_in_primary_mode_outputs_seven_channels_and_trains(self) -> None:
        model = PengwinMultiTaskModel(base_channels=4, instance_mode="primary")
        labels = _synthetic_labels()
        outputs = model(torch.rand(2, 1, 64, 64))
        self.assertEqual(outputs["instance"].shape[1], INSTANCE_MODES["primary"])
        loss, parts = multitask_loss(outputs, labels_to_grid(labels), labels,
                                     weights=MultiTaskLossWeights(fracture_weight=4.0))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None for p in model.segmentation_head.instance.parameters()))

    def test_ids_mode_is_unchanged(self) -> None:
        model = PengwinMultiTaskModel(base_channels=4)
        outputs = model(torch.rand(1, 1, 64, 64))
        self.assertEqual(outputs["instance"].shape[1], 31)

    def test_sagittal_flip_swaps_hipbones_and_keeps_sacrum(self) -> None:
        labels = _synthetic_labels()
        images = torch.rand(2, 1, 64, 64)
        flipped_images, flipped_labels = flip_sagittal(images, labels)
        self.assertTrue(torch.equal(flipped_images[..., 0], images[..., -1]))
        self.assertEqual(int(flipped_labels[0, 50, 63 - 10]), 21)   # 11 → 21 espejado
        self.assertEqual(int(flipped_labels[0, 50, 63 - 29]), 22)   # 12 → 22
        self.assertEqual(int(flipped_labels[0, 50, 63 - 50]), 11)   # 21 → 11
        self.assertEqual(int(flipped_labels[0, 20, 63 - 20]), 1)    # sacro intacto
        # Doble flip devuelve el original.
        back_images, back_labels = flip_sagittal(flipped_images, flipped_labels)
        self.assertTrue(torch.equal(back_labels, labels))
        self.assertTrue(torch.equal(back_images, images))

    def test_fracture_weight_peaks_at_fragment_interface(self) -> None:
        labels = _synthetic_labels()
        weight = fracture_distance_weight(labels, strength=4.0, sigma=3.0)
        self.assertGreater(float(weight[0, 30, 20]), 4.5)   # interfaz 1|2
        self.assertLess(float(weight[0, 3, 3]), 1.05)       # lejos de cualquier fractura
        self.assertTrue((fracture_distance_weight(labels, strength=0.0) == 1).all())


if __name__ == "__main__":
    unittest.main()


class RegionCropTests(unittest.TestCase):
    def test_crops_one_per_region_and_keep_only_region_ids(self) -> None:
        from week10_multitask import region_crops
        labels = _synthetic_labels()
        images = torch.rand(2, 1, 64, 64)
        crops, crop_labels, boxes = region_crops(images, labels)
        self.assertEqual(len(boxes), 6)  # 3 regiones × 2 imágenes
        self.assertEqual(tuple(crops.shape), (6, 1, 64, 64))
        for (bi, region, *_), lab in zip(boxes, crop_labels):
            start = 10 * (region - 1) + 1
            ids = [v for v in torch.unique(lab).tolist() if v > 0]
            self.assertTrue(all(start <= v <= start + 9 for v in ids))
            self.assertIn(start, ids)

    def test_two_pass_returns_full_size_map_with_region_classes(self) -> None:
        from week10_multitask import two_pass_instances, labels_to_semantic
        model = PengwinMultiTaskModel(base_channels=4, instance_mode="primary").eval()
        labels = _synthetic_labels()
        images = torch.rand(2, 1, 64, 64)
        out = two_pass_instances(model, images, labels_to_semantic(labels))
        self.assertEqual(tuple(out.shape), (2, 64, 64))
        sem = labels_to_semantic(labels)
        for region in (1, 2, 3):
            inside = sem == region
            values = torch.unique(out[inside]).tolist()
            self.assertTrue(all(v in (2 * region - 1, 2 * region) for v in values))
        self.assertTrue((out[sem == 0] == out[sem == 0]).all())
