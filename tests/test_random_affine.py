import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_s10 import random_affine  # noqa: E402


def _batch():
    y = torch.zeros(6, 96, 96, dtype=torch.long)
    y[:, 20:60, 15:45] = 1
    y[:, 30:40, 45:52] = 2
    y[:, 50:80, 55:85] = 21
    return (y > 0).float()[:, None], y


def test_sin_probabilidad_no_cambia_nada():
    x, y = _batch()
    x2, y2 = random_affine(x, y, probability=0.0)
    assert torch.equal(y2, y) and torch.allclose(x2, x, atol=1e-5)


def test_etiquetas_siguen_siendo_ids_validos_y_alineadas_con_la_imagen():
    torch.manual_seed(0)
    x, y = _batch()
    x2, y2 = random_affine(x, y, probability=1.0)
    assert x2.shape == x.shape and y2.shape == y.shape and y2.dtype == torch.long
    assert set(y2.unique().tolist()) <= {0, 1, 2, 21}        # vecino más cercano: no inventa IDs
    assert not torch.equal(y2, y)                             # sí transformó
    agreement = ((x2[:, 0] > 0.5) == (y2 > 0)).float().mean()
    assert agreement > 0.97                                   # imagen y etiqueta se mueven juntas


def test_fuerza_cero_es_identidad():
    x, y = _batch()
    x2, y2 = random_affine(x, y, strength=0.0, probability=1.0)
    assert torch.equal(y2, y)


def test_spacing_isotropico_reproduce_el_aumento_anterior():
    x, y = _batch()
    torch.manual_seed(7)
    old_x, old_y = random_affine(x, y, probability=1.0)
    torch.manual_seed(7)
    spacing = torch.ones(len(x), 2)
    mm_x, mm_y = random_affine(x, y, probability=1.0, spacing_yx=spacing)
    assert torch.allclose(mm_x, old_x, atol=1e-6) and torch.equal(mm_y, old_y)


def test_spacing_no_cuadrado_cambia_la_matriz_sin_inventar_ids():
    x, y = _batch()
    torch.manual_seed(11)
    pixel_x, _ = random_affine(x, y, probability=1.0)
    torch.manual_seed(11)
    physical_x, physical_y = random_affine(
        x, y, probability=1.0, spacing_yx=torch.tensor([[1.8, 0.6]]).repeat(len(x), 1))
    assert not torch.allclose(physical_x, pixel_x)
    assert set(physical_y.unique().tolist()) <= {0, 1, 2, 21}
