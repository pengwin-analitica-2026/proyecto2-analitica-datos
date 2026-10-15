"""Subida de un CT propio: lectura de .mha, NIfTI y series DICOM, orientación y exportación sin GT."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orientacion import flip_inplane  # noqa: E402
from volumen_entrada import huella, leer_subida  # noqa: E402

LPI = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


def volumen_prueba(z=12, y=20, x=24) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(-1000, 1500, (z, y, x)).astype(np.int16)


def a_bytes(array: np.ndarray, extension: str, direccion=None, spacing=(0.8, 0.9, 1.5)) -> bytes:
    imagen = sitk.GetImageFromArray(array)
    imagen.SetSpacing(spacing)
    if direccion is not None:
        imagen.SetDirection(direccion)
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / f"v{extension}"
        sitk.WriteImage(imagen, str(ruta))
        return ruta.read_bytes()


def serie_dicom(array: np.ndarray, spacing=(0.7, 0.7, 2.0)) -> list[tuple[str, bytes]]:
    """Un archivo DICOM por corte, como los guarda un tomógrafo (desordenados a propósito)."""
    archivos = []
    escritor = sitk.ImageFileWriter()
    escritor.KeepOriginalImageUIDOn()
    serie = "1.2.826.0.1.3680043.2.1125.1.123456"
    with tempfile.TemporaryDirectory() as d:
        for z in range(len(array)):
            corte = sitk.GetImageFromArray(array[z:z + 1])
            corte.SetSpacing(spacing)
            for tag, valor in (("0008|0060", "CT"), ("0020|000e", serie), ("0020|000d", serie + ".0"),
                               ("0008|0018", f"{serie}.{z + 1}"), ("0020|0013", str(z + 1)),
                               ("0020|0032", f"0\\0\\{z * spacing[2]}"),
                               ("0020|0037", "1\\0\\0\\0\\1\\0"), ("0018|0050", str(spacing[2])),
                               ("0028|0030", f"{spacing[1]}\\{spacing[0]}")):
                corte.SetMetaData(tag, valor)
            ruta = Path(d) / f"IM{z:04d}.dcm"
            escritor.SetFileName(str(ruta))
            escritor.Execute(corte)
            archivos.append((f"paciente_x/IM{z:04d}.dcm", ruta.read_bytes()))
    return archivos[::-1]


class LeerSubida(unittest.TestCase):
    def test_mha_lpi_se_endereza_como_to_canonical(self):
        array = volumen_prueba()
        vol = leer_subida([("caso nuevo.mha", a_bytes(array, ".mha", LPI))])
        np.testing.assert_array_equal(vol.image, flip_inplane(array))
        self.assertEqual(vol.spacing_xyz, (0.8, 0.9, 1.5))
        self.assertEqual(vol.nombre, "caso_nuevo")
        self.assertFalse(vol.orientacion["estandar"])

    def test_mha_estandar_queda_igual(self):
        array = volumen_prueba()
        vol = leer_subida([("a.mha", a_bytes(array, ".mha"))])
        np.testing.assert_array_equal(vol.image, array)
        self.assertTrue(vol.orientacion["estandar"])

    def test_nifti_comprimido(self):
        array = volumen_prueba()
        vol = leer_subida([("a.nii.gz", a_bytes(array, ".nii.gz"))])
        self.assertEqual(vol.image.shape, array.shape)
        self.assertEqual(vol.fuente, "archivo .nii.gz")

    def test_serie_dicom_desordenada(self):
        array = volumen_prueba(z=10)
        vol = leer_subida(serie_dicom(array))
        np.testing.assert_array_equal(vol.image, array)          # GDCM ordena por posición
        np.testing.assert_allclose(vol.spacing_xyz, (0.7, 0.7, 2.0), atol=1e-4)
        self.assertEqual(vol.fuente, "serie DICOM (10 cortes)")
        self.assertEqual(vol.nombre, "paciente_x")

    def test_png_y_vacio_dan_error_claro(self):
        with self.assertRaisesRegex(ValueError, "Hounsfield"):
            leer_subida([("c/001.png", b"x"), ("c/002.png", b"y")])
        with self.assertRaises(ValueError):
            leer_subida([])
        with self.assertRaisesRegex(ValueError, "serie DICOM"):
            leer_subida([("c/notas.txt", b"hola")])

    def test_pocos_cortes(self):
        with self.assertRaisesRegex(ValueError, "cortes"):
            leer_subida([("a.mha", a_bytes(volumen_prueba(z=3), ".mha"))])

    def test_huella_estable(self):
        a = [("x.mha", b"123"), ("y.raw", b"456")]
        self.assertEqual(huella(a), huella(a[::-1]))
        self.assertNotEqual(huella(a), huella([("x.mha", b"124")]))


class ExportarVolumenSinGroundTruth(unittest.TestCase):
    def test_pipeline_completo_con_predictor_falso(self):
        from demo_export import exportar_volumen
        size = 64
        image = np.full((12, 48, 48), -1000, np.int16)
        image[:, 10:30, 10:20] = 700                      # "hueso"

        def predecir(images, progress=None):
            n = len(images)
            semantic = np.zeros((n, size, size), np.uint8)
            ps = np.zeros_like(semantic)
            semantic[2:10, 10:30, 10:20] = 2
            ps[2:10, 10:30, 10:16] = 3                    # primario del coxal izquierdo
            ps[2:10, 10:30, 18:20] = 4                    # secundario, separado por un hueco
            return {"semantic": semantic, "primary_secondary": ps,
                    "classification": np.zeros((n, 3), np.float32),
                    "boxes_raw": [[] for _ in range(n)], "boxes_refined": [[] for _ in range(n)],
                    "seconds": 0.1}

        info = {"checkpoint": "falso.pth", "parameters": 1, "tta": False, "canonical": True, "device": "cpu"}
        with tempfile.TemporaryDirectory() as d:
            out = exportar_volumen(image, (0.8, 0.8, 1.0), Path(d) / "subido", info, predecir, nombre="subido",
                                   size=size, enderezar=False, umbral_mm=False, min_voxels=5,
                                   extra_meta={"subido": {"fuente": "prueba"}}, log=lambda m: None)
            meta = json.loads((out / "meta.json").read_text())
            self.assertIsNone(meta["fragments_gt"])
            self.assertIsNone(meta["comparison"])
            self.assertEqual(meta["subido"]["fuente"], "prueba")
            self.assertEqual(meta["model"]["checkpoint"], "falso.pth")
            ids = {r["id"]: r for r in meta["fragments_pred"]}
            self.assertIn(11, ids)
            self.assertGreater(ids[12]["distance_mm"], 0)
            with np.load(out / "volume.npz") as v:
                self.assertNotIn("gt_ids", v.files)
                self.assertEqual(v["images"].shape, (12, size, size))
            self.assertTrue((out / "meshes.npz").exists())   # raw_view.npz puede faltar: el V1 es opcional


if __name__ == "__main__":
    unittest.main()
