import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orientacion import (LPI_CASES, analyse_direction, canonicalize_arrays,
                         estimate_pelvis_correction, from_canonical, is_rotated,
                         rotate_inplane_physical, to_canonical)  # noqa: E402


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


def test_header_is_rotated_lee_la_cabecera(tmp_path):
    from orientacion import flip_inplane, header_is_rotated
    lpi, rai, sin = tmp_path / "lpi.mha", tmp_path / "rai.mha", tmp_path / "sin.mha"
    lpi.write_bytes(b"ObjectType = Image\nTransformMatrix = -1 0 0 0 -1 0 0 0 1\nElementDataFile = LOCAL\n\x00\x01")
    rai.write_bytes(b"ObjectType = Image\nTransformMatrix = 1 0 0 0 1 0 0 0 1\nElementDataFile = LOCAL\n\x00\x01")
    sin.write_bytes(b"ObjectType = Image\nElementDataFile = LOCAL\n")
    assert header_is_rotated(lpi) is True
    assert header_is_rotated(rai) is False
    assert header_is_rotated(sin) is None
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    np.testing.assert_array_equal(flip_inplane(flip_inplane(volume)), volume)
    np.testing.assert_array_equal(flip_inplane(volume)[:, 0, 0], volume[:, -1, -1])


def test_productos_punto_y_cruz_validan_la_direccion():
    info = analyse_direction(np.diag([-1.0, -1.0, 1.0]))
    assert info["orthonormal"] and info["axis_aligned"]
    np.testing.assert_allclose(info["dots"], np.eye(3))
    np.testing.assert_allclose(info["cross_xy"], [0, 0, 1])
    assert info["handedness"] == pytest.approx(1.0)

    invalid = analyse_direction(np.array([[1, 0.2, 0], [0, 1, 0], [0, 0, 1]]))
    assert not invalid["orthonormal"]


def test_reorientacion_por_metadatos_coincide_con_el_giro_conocido():
    volume = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    label = (volume % 5).astype(np.uint8)
    out, out_label, spacing, info = canonicalize_arrays(
        volume, label, np.diag([-1.0, -1.0, 1.0]), (0.7, 0.8, 1.2))
    np.testing.assert_array_equal(out, volume[:, ::-1, ::-1])
    np.testing.assert_array_equal(out_label, label[:, ::-1, ::-1])
    assert spacing == pytest.approx((0.7, 0.8, 1.2)) and info["applied"]


def test_rotacion_fisica_recupera_un_giro_conocido_sin_tocar_ids():
    z, y, x = np.indices((12, 96, 96))
    volume = np.zeros((12, 96, 96), dtype=np.uint8)
    pelvis = ((((x - 28) / 13) ** 2 + ((y - 48) / 25) ** 2 < 1) |
              (((x - 68) / 13) ** 2 + ((y - 48) / 25) ** 2 < 1) |
              (((x - 48) / 7) ** 2 + ((y - 70) / 11) ** 2 < 1))
    volume[pelvis] = 180
    turned = rotate_inplane_physical(volume, 12.0, 0.8, 0.8, order=1)
    info = estimate_pelvis_correction(turned, 0.8, 0.8)
    assert info["accepted"] and info["angle_degrees"] == pytest.approx(-12.0, abs=1.0)

    labels = (volume > 0).astype(np.uint8) * 21
    labels = rotate_inplane_physical(labels, 17.0, 0.8, 1.3, order=0)
    assert set(np.unique(labels)).issubset({0, 21})
