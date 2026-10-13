import numpy as np

from src.evaluar_metricas_superficie import summarize
from src.volume_postprocess import fragment_surface_table


def test_surface_table_reports_all_gt_fragments_and_missing_coverage():
    truth = np.zeros((8, 12, 12), dtype=np.uint8)
    prediction = np.zeros_like(truth)
    truth[1:4, 1:5, 1:5] = 1
    truth[4:7, 7:10, 7:10] = 2
    prediction[1:4, 1:5, 2:6] = 1

    rows = fragment_surface_table(prediction, truth, (1.0, 1.0, 1.0))
    assert len(rows) == 2
    principal, secondary = rows
    assert principal["tipo"] == "principal"
    assert principal["detectado"] is True
    assert principal["assd_mm"] > 0
    assert principal["hd95_mm"] > 0
    assert secondary["tipo"] == "secundario"
    assert secondary["detectado"] is False
    assert secondary["assd_mm"] is None
    assert secondary["hd95_mm"] is None

    summary = summarize(rows)
    assert summary["n_fragmentos_gt"] == 2
    assert summary["n_fragmentos_detectados"] == 1
    assert summary["cobertura_fragmentos"] == 0.5


def test_surface_table_uses_physical_spacing():
    truth = np.zeros((5, 7, 7), dtype=np.uint8)
    prediction = np.zeros_like(truth)
    truth[1:4, 2:5, 2:5] = 1
    prediction[2:5, 2:5, 2:5] = 1

    isotropic = fragment_surface_table(prediction, truth, (1.0, 1.0, 1.0))[0]
    thick_z = fragment_surface_table(prediction, truth, (1.0, 1.0, 3.0))[0]
    assert thick_z["assd_mm"] > isotropic["assd_mm"]
    assert thick_z["hd95_mm"] > isotropic["hd95_mm"]
