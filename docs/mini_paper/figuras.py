"""Figuras del mini-paper que salen de los JSON de ``outputs/s10`` (no entrena ni infiere).

Uso, desde la raíz del repositorio:  python docs/mini_paper/figuras.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

import resumen_s10 as r  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "fig"
SALIDA.mkdir(exist_ok=True)

fig = r.figura_por_caso(("D5 (primera evaluación)", "TEST_d5_s42_tta"), ("H6 (segunda evaluación)", "TEST2_h6_s42_tta"))
fig.set_size_inches(4.6, 4.0)      # tamaño pensado para ir a media página
fig.axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")   # arriba a la izquierda no hay puntos
fig.tight_layout()
fig.savefig(SALIDA / "por_caso_test.png", dpi=220)

fig = r.figura_estres([
    ("D5", [(0, "d5_ema_w32_e60_s42_val"), (30, "estres2_d5_rot_30"), (60, "estres_d5_rot60")]),
    ("H5", [(0, "h5_auggeo_canon_ema_w32_e60_s42_val"), (30, "estres2_h5_rot_30"), (60, "estres2_h5_rot_60")]),
    ("H6", [(0, "h6_auggeo_rot35_canon_ema_w32_e60_s42_val"), (30, "estres2_h6_rot_30"), (60, "estres2_h6_rot_60")]),
])
fig.set_size_inches(4.6, 4.0)
fig.tight_layout()
fig.savefig(SALIDA / "estres_giro.png", dpi=220)
