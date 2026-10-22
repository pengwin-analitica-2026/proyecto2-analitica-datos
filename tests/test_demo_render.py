import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demo_render import (distance_text, fragment_rgb, local_name, mesh_figure, region_of, render_slice,
                         slice_labels)


def rows():
    return [{"id": 1, "region": "sacro", "is_main": True, "voxels": 100, "volume_mm3": 1.0,
             "distance_mm": None, "centroid_zyx_vox": [2.0, 10.0, 10.0], "z_range": [0, 5]},
            {"id": 2, "region": "sacro", "is_main": False, "voxels": 20, "volume_mm3": 1.0,
             "distance_mm": 12.84, "centroid_zyx_vox": [2.0, 10.0, 40.0], "z_range": [0, 5]},
            {"id": 12, "region": "coxal_izquierdo", "is_main": False, "voxels": 20, "volume_mm3": 1.0,
             "distance_mm": 0.0, "centroid_zyx_vox": [2.0, 30.0, 40.0], "z_range": [0, 5]}]


class Render(unittest.TestCase):
    def test_names_regions_and_colors(self):
        self.assertEqual([local_name(i) for i in (1, 2, 11, 13, 21)], ["principal", "F1", "principal", "F2", "principal"])
        self.assertEqual([region_of(i) for i in (1, 10, 11, 20, 21, 30)],
                         ["sacro", "sacro", "coxal_izquierdo", "coxal_izquierdo", "coxal_derecho", "coxal_derecho"])
        colors = {fragment_rgb(i) for i in (1, 2, 3, 11, 12, 21, 22)}
        self.assertEqual(len(colors), 7)                                   # todos distinguibles
        self.assertNotEqual(fragment_rgb(1), fragment_rgb(11))

    def test_distance_text(self):
        self.assertEqual(distance_text(12.84), "12.8 mm")
        self.assertEqual(distance_text(0.0), "0.0 mm (contacto)")
        self.assertEqual(distance_text(None), "sin medir")

    def test_labels_only_for_secondary_fragments_present_in_the_slice(self):
        ids = np.zeros((64, 64), np.uint8)
        ids[10:30, 5:25] = 1
        ids[10:20, 40:50] = 2
        labels = slice_labels(ids, rows())
        self.assertEqual([l["id"] for l in labels], [2])                  # el 12 no está en este corte
        self.assertEqual(labels[0]["text"], "F1 · 12.8 mm")        # corto: siempre en mm
        self.assertAlmostEqual(labels[0]["x"], 44.5)

    def test_render_slice_draws_boxes_masks_and_labels(self):
        image = np.full((64, 64), 40, np.uint8)
        ids = np.zeros((64, 64), np.uint8)
        ids[10:30, 5:25] = 1
        ids[10:20, 40:50] = 2
        meta = {"boxes_refined": [{"region": "sacro", "region_id": 0, "score": 0.9, "box": [0.05, 0.1, 0.8, 0.5]}],
                "boxes_raw": []}
        plain = render_slice(image, ids, rows(), meta, show_masks=False, show_boxes=False, scale=4)
        full = render_slice(image, ids, rows(), meta, scale=4)
        self.assertEqual(full.size, (256, 256))
        self.assertGreater(int(np.abs(np.asarray(full).astype(int) - np.asarray(plain).astype(int)).sum()), 0)
        masks_only = render_slice(image, ids, rows(), None, show_boxes=False, scale=4)
        self.assertEqual(masks_only.size, (256, 256))

    def test_mesh_figure_has_one_mesh_per_fragment_and_a_label_trace(self):
        verts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
        meshes = {f"p_{i}_{k}": a for i in (1, 2, 12) for k, a in (("v", verts), ("f", faces))}
        fig = mesh_figure(meshes, rows(), (1.0, 1.0, 1.0))
        kinds = [t.type for t in fig.data]
        self.assertEqual(kinds.count("mesh3d"), 3)
        self.assertEqual(kinds.count("scatter3d"), 1)
        texts = [a.text for a in fig.layout.scene.annotations]
        self.assertEqual(len(texts), 2)
        self.assertTrue(any("F1 · 12.8 mm" in text for text in texts))
        self.assertTrue(any("F1 · 0.0 mm" in text for text in texts))
        separated = mesh_figure(meshes, rows(), (1.0, 1.0, 1.0), label_contacts=False)
        self.assertEqual(len(separated.layout.scene.annotations), 1)
        no_main = mesh_figure(meshes, rows(), (1.0, 1.0, 1.0), show_main=False)
        self.assertEqual([t.type for t in no_main.data].count("mesh3d"), 2)



class VentanaVisual(unittest.TestCase):
    def test_deshace_la_ventana_del_modelo(self):
        from demo_render import lut_ventana
        from pengwin_io import window_hu
        hu = np.array([-500, -160, 0, 240, 600, 1300], dtype=np.float32)
        gris = np.rint(window_hu(hu) * 255).astype(np.uint8)       # como se guarda el corte
        visto = lut_ventana(-160, 240)[gris].astype(float)
        esperado = np.clip(255 * (hu + 160) / 400, 0, 255)
        np.testing.assert_allclose(visto, esperado, atol=4)       # error de redondeo de uint8

    def test_sin_ventana_no_cambia_el_corte(self):
        imagen = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
        ids = np.zeros((64, 64), dtype=np.uint8)
        normal = np.asarray(render_slice(imagen, ids, [], show_boxes=False, scale=1))
        otra = np.asarray(render_slice(imagen, ids, [], show_boxes=False, scale=1, window=(150, 1300)))
        self.assertFalse(np.array_equal(normal, otra))
        self.assertEqual(normal.shape, otra.shape)


if __name__ == "__main__":
    unittest.main()
