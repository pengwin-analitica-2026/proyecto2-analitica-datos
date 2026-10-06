import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orientacion import LPI_CASES, from_canonical, is_rotated, to_canonical  # noqa: E402


def test_gira_180_solo_los_casos_lpi_y_es_su_propia_inversa():
    volume = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    assert to_canonical(volume, "001") is volume                    # RAI: intacto
    turned = to_canonical(volume, "044")                            # LPI: filas y columnas invertidas
    assert np.array_equal(turned, volume[:, ::-1, ::-1]) and turned.flags["C_CONTIGUOUS"]
    assert np.array_equal(from_canonical(turned, 44), volume)


def test_la_lista_tiene_34_casos():
    assert len(LPI_CASES) == 34 and is_rotated(87) and not is_rotated("012")


def test_la_lista_coincide_con_las_cabeceras_si_estan_los_datos():
    files = sorted((ROOT / "data" / "raw").glob("PENGWIN_CT_train_images_part*/*.mha"))
    if not files:
        pytest.skip("data/raw no está en esta máquina")
    for path in files:
        with open(path, "rb") as stream:
            lpi = b"AnatomicalOrientation = LPI" in stream.read(2048)
        assert lpi == is_rotated(path.stem), path.name
