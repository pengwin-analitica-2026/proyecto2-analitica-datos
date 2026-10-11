"""
hu_windowing_check.py
----------------------
Aplica el ventaneo HU (con clip previo por el padding/metal detectado) sobre
varios casos de ejemplo y guarda una imagen comparativa (crudo vs ventaneado)
para verificar visualmente que el hueso queda bien contrastado.

Requisitos:
    pip install SimpleITK numpy matplotlib

Uso:
    python hu_windowing_check.py
"""

from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
DATA_ROOT = Path("data/raw")
IMAGES_DIRS = [
    DATA_ROOT / "PENGWIN_CT_train_images_part1",
    DATA_ROOT / "PENGWIN_CT_train_images_part2",
]

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Casos de ejemplo a revisar: uno normal y dos con valores extremos conocidos
CASOS_EJEMPLO = ["001", "080", "068"]

# Clip previo para neutralizar padding (-2048/-1023...) y metal (picos >5000 HU)
CLIP_MIN, CLIP_MAX = -1024, 3000

# Ventana ósea (ajusta si al ver el resultado necesitas más/menos contraste)
WINDOW_CENTER, WINDOW_WIDTH = 400, 1800


def find_image_path(case_id):
    for img_dir in IMAGES_DIRS:
        candidate = img_dir / f"{case_id}.mha"
        if candidate.exists():
            return candidate
    return None


def apply_hu_window(volume, center, width, clip_min, clip_max):
    volume = np.clip(volume, clip_min, clip_max)
    low, high = center - width / 2, center + width / 2
    windowed = np.clip(volume, low, high)
    windowed = (windowed - low) / (high - low)
    return windowed


def main():
    for case_id in CASOS_EJEMPLO:
        img_path = find_image_path(case_id)
        if img_path is None:
            print(f"[AVISO] No se encontró el caso {case_id}, se omite.")
            continue

        img = sitk.ReadImage(str(img_path))
        volume = sitk.GetArrayFromImage(img)  # (z, y, x)

        corte_central = volume.shape[0] // 2
        corte_crudo = volume[corte_central]
        corte_ventaneado = apply_hu_window(
            corte_crudo, WINDOW_CENTER, WINDOW_WIDTH, CLIP_MIN, CLIP_MAX
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(corte_crudo, cmap="gray")
        axes[0].set_title(f"Caso {case_id} — HU crudo")
        axes[0].axis("off")

        axes[1].imshow(corte_ventaneado, cmap="gray")
        axes[1].set_title(f"Caso {case_id} — ventaneado (clip {CLIP_MIN}/{CLIP_MAX}, W{WINDOW_WIDTH}/C{WINDOW_CENTER})")
        axes[1].axis("off")

        out_path = OUTPUT_DIR / f"hu_window_check_{case_id}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
