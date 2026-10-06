"""Pruebas del caché por caso (volumen completo) con un volumen sintético."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_case_cache as bcc
from pengwin_io import resize_label_slice, resize_slice, window_hu


def write_case(root: Path, case_id: str, image: np.ndarray, label: np.ndarray,
               spacing=(0.8, 0.8, 1.25)) -> None:
    for folder, array in (("PENGWIN_CT_train_images_part1", image), ("PENGWIN_CT_train_labels", label)):
        (root / folder).mkdir(parents=True, exist_ok=True)
        itk = sitk.GetImageFromArray(array)
        itk.SetSpacing(spacing)
        sitk.WriteImage(itk, str(root / folder / f"{case_id}.mha"))


class BuildCaseCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "raw"
        self.out = Path(self.tmp.name) / "out"
        rng = np.random.default_rng(0)
        # Volumen no cuadrado (como el caso 039: 154x322) con hueso solo en z = 2..5.
        self.image = rng.integers(-1000, 1500, size=(8, 40, 80)).astype(np.int16)
        self.label = np.zeros((8, 40, 80), dtype=np.uint8)
        self.label[2:6, 10:30, 10:35] = 1
        self.label[2:6, 10:30, 45:70] = 11
        self.label[3:5, 22:28, 60:68] = 12
        write_case(self.root, "001", self.image, self.label)
        self.row = bcc.build_case("001", str(self.out), (32,), 1.5, "train",
                                  raw_root=str(self.root), modes=("stretch", "pad"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_row_describes_the_volume(self):
        self.assertEqual(self.row["n_slices"], 8)
        self.assertEqual(self.row["n_bone_slices"], 4)
        self.assertEqual((self.row["first_bone_z"], self.row["last_bone_z"]), (2, 5))
        self.assertEqual(self.row["n_fragments"], 3)
        self.assertAlmostEqual(self.row["spacing_z_mm"], 1.25)

    def test_stretch_matches_pengwin_io_up_to_uint8_rounding(self):
        cached = bcc.load_case_cache(self.out / "cache_32", "001")
        self.assertEqual(cached["images"].shape, (8, 32, 32))
        self.assertEqual(cached["images"].dtype, np.uint8)
        for z in (0, 3, 7):
            reference = resize_slice(window_hu(self.image[z], gamma=1.5), 32)
            error = np.abs(bcc.from_uint8(cached["images"][z]) - reference).max()
            self.assertLessEqual(error, 0.5 / 255 + 1e-6)
            np.testing.assert_array_equal(cached["labels"][z], resize_label_slice(self.label[z], 32))

    def test_every_slice_is_kept_including_those_without_bone(self):
        cached = bcc.load_case_cache(self.out / "cache_32", "001")
        has_bone = (cached["labels"] != 0).any(axis=(1, 2))
        np.testing.assert_array_equal(has_bone, [False, False, True, True, True, True, False, False])

    def test_pad_mode_keeps_aspect_ratio(self):
        cached = bcc.load_case_cache(self.out / "cache_32_pad", "001")
        self.assertEqual(tuple(cached["pad_yx"]), (20, 0))  # 40x80 -> 80x80
        region = cached["labels"][3] == 1
        ys, xs = np.where(region)
        height, width = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        # La región original mide 20x25; con relleno se conserva la proporción 0.8.
        self.assertAlmostEqual(height / width, 20 / 25, delta=0.15)
        # En modo stretch la misma región queda deformada (40x80 -> 32x32 duplica el alto relativo).
        stretched = bcc.load_case_cache(self.out / "cache_32", "001")["labels"][3] == 1
        ys, xs = np.where(stretched)
        self.assertGreater((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1), 1.3)
        # El relleno es fondo: las filas superiores no tienen etiqueta ni señal.
        self.assertEqual(int(cached["labels"][3][:6].max()), 0)
        self.assertEqual(int(cached["images"][3][:6].max()), 0)

    def test_native_label_is_stored_unchanged(self):
        with np.load(self.out / "labels_native" / "001.npz") as data:
            np.testing.assert_array_equal(data["label"], self.label)
            np.testing.assert_allclose(data["spacing_xyz"], (0.8, 0.8, 1.25))

    def test_index_file_accumulates_cases(self):
        bcc.write_index([self.row], self.out)
        other = dict(self.row, case_id="002", split="val")
        path = bcc.write_index([other], self.out)
        lines = path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("001,train"))
        self.assertTrue(lines[2].startswith("002,val"))


if __name__ == "__main__":
    unittest.main()
