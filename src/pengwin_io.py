"""
pengwin_io.py
-------------
Modulo unico de carga y preprocesamiento del dataset PENGWIN.

Centraliza en un solo lugar lo que antes estaba repetido en
build_manifest.py, mip_visualizer.py y hu_windowing_check.py, para que
entrenamiento, inferencia y los tres visualizadores usen exactamente la
misma logica de carga, clip, ventaneo y resize (pipeline identico en
train y en inferencia, como pidio el profesor).

Convenciones del equipo (mismas constantes que el resto del repo):
- Clip previo: [-1024, 3000] HU, neutraliza padding y metal.
- Ventana osea de visualizacion/entrada al modelo: centro 400, ancho 1800.
- CLAHE (clip_limit=0.02) sobre la imagen ya en [0,1], activado por defecto
  en window_hu(): corrige casos con histograma oseo corrido/comprimido
  (ej. caso 025, diagnosticado en asesoria con el profesor), sin tocar los
  umbrales en HU crudo que usan bone_mask/body_mask.
- Umbral osea (mascara gruesa, sin ML): 250 HU sobre el volumen crudo.

Uso:
    from pengwin_io import load_case, window_hu, resize_slice

    caso = load_case("001")
    corte = window_hu(caso["image"][caso["image"].shape[0] // 2])
    corte_256 = resize_slice(corte, 256)
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.transform import resize as _sk_resize
from skimage import exposure as _sk_exposure

# ---------------------------------------------------------------------------
# CONFIGURACION — mismas rutas y constantes que el resto del repo
# ---------------------------------------------------------------------------
DATA_ROOT = Path("data/raw")
IMAGES_DIRS = [
    DATA_ROOT / "PENGWIN_CT_train_images_part1",
    DATA_ROOT / "PENGWIN_CT_train_images_part2",
]
LABELS_DIR = DATA_ROOT / "PENGWIN_CT_train_labels"

CLIP_MIN, CLIP_MAX = -1024, 3000        # neutraliza padding y metal
WINDOW_CENTER, WINDOW_WIDTH = 400, 1800  # ventana osea
BONE_HU_THRESHOLD = 250                  # umbral osea (mascara gruesa, no-ML)

# Fragmento principal por convencion del dataset: la etiqueta que termina en
# 1 (1, 11, 21) es el fragmento mayor de su region. Se usa para la distancia
# de separacion; se valida por separado en el EDA (fragmento_principal.py).
LABEL_TO_REGION = {1: "sacro", 11: "coxal_izq", 21: "coxal_der"}


# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
def find_image_path(case_id: str) -> Path | None:
    """Busca el .mha de imagen de un caso en part1/part2."""
    case_id = str(case_id).zfill(3)
    for img_dir in IMAGES_DIRS:
        candidate = img_dir / f"{case_id}.mha"
        if candidate.exists():
            return candidate
    return None


def find_label_path(case_id: str) -> Path | None:
    """Busca el .mha de label de un caso."""
    case_id = str(case_id).zfill(3)
    candidate = LABELS_DIR / f"{case_id}.mha"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# CARGA
# ---------------------------------------------------------------------------
@dataclass
class PengwinCase:
    case_id: str
    image: np.ndarray          # (z, y, x) HU crudo, int
    label: np.ndarray | None   # (z, y, x) etiquetas de fragmento, o None
    spacing_xyz: tuple         # (sx, sy, sz) mm/voxel, del header SimpleITK


def load_case(case_id: str, load_label: bool = True) -> PengwinCase:
    """
    Carga imagen (+ label si existe) de un caso y devuelve el volumen crudo
    en HU junto con el spacing fisico del header. No aplica clip ni ventaneo:
    eso lo hacen clip_hu / window_hu, para poder medir en HU crudo cuando
    haga falta (ej. EDA de intensidades).
    """
    img_path = find_image_path(case_id)
    if img_path is None:
        raise FileNotFoundError(f"No se encontro imagen para el caso {case_id}")

    img = sitk.ReadImage(str(img_path))
    image = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing_xyz = img.GetSpacing()       # SimpleITK: (x, y, z) en mm

    label = None
    if load_label:
        label_path = find_label_path(case_id)
        if label_path is not None:
            lbl_img = sitk.ReadImage(str(label_path))
            label = sitk.GetArrayFromImage(lbl_img)

    return PengwinCase(case_id=str(case_id).zfill(3), image=image, label=label,
                        spacing_xyz=spacing_xyz)


# ---------------------------------------------------------------------------
# PREPROCESAMIENTO: clip, ventaneo, resize
# ---------------------------------------------------------------------------
def clip_hu(volume: np.ndarray, clip_min: int = CLIP_MIN, clip_max: int = CLIP_MAX) -> np.ndarray:
    """Clip previo en HU crudo. Neutraliza padding (-2048/-1023) y metal (picos >5000 HU)."""
    return np.clip(volume, clip_min, clip_max)


CLAHE_CLIP_LIMIT = 0.02  # mismo valor con el que se comparo contra gamma y ecualizacion global


def window_hu(volume: np.ndarray, center: int = WINDOW_CENTER, width: int = WINDOW_WIDTH,
              clip_min: int = CLIP_MIN, clip_max: int = CLIP_MAX,
              aplicar_clahe: bool = True) -> np.ndarray:
    """
    Ventaneo HU -> [0, 1] float32 + CLAHE por defecto. Aplica clip_hu
    internamente, asi que se puede llamar directo sobre el volumen crudo
    de load_case(). Espera un CORTE 2D (y, x), no un volumen 3D completo
    — asi la llaman todos los scripts del repo (train_overfit_detection.py,
    prepare_week9_detection_data.py); CLAHE es local por parches y no tiene
    sentido aplicarlo de una sola vez sobre un volumen 3D entero.

    CLAHE (skimage.exposure.equalize_adapthist) se agrego tras diagnosticar
    el caso 025 en la asesoria con el profesor: NO es metal (esa hipotesis
    se descarto, ver docs/resumen_semana8.md) sino que su histograma oseo
    esta corrido y comprimido ~150-200 HU por debajo de lo tipico (unico
    caso con z-score < -2 en hueso_media/hueso_p50 sobre los 100 casos),
    probablemente por diferencia de kernel de reconstruccion entre
    instituciones. Se comparo contra gamma fijo (distorsiona por igual los
    casos ya normales, descartado) y ecualizacion global (demasiado
    agresiva, aplana el tejido blando a un bloque gris uniforme y pierde
    toda la anatomia, descartada). CLAHE es local/adaptivo por parches:
    en el caso 025 recupera la textura trabecular que el corrimiento HU
    escondia, y en casos ya normales no distorsiona la imagen de forma
    perceptible. Comparacion visual en
    outputs/eda/diagnostico_normalizacion_caso025.png.

    Se aplica AQUI (a la imagen ya en [0,1]), nunca al HU crudo: bone_mask,
    body_mask y clean_bone_mask siguen operando sobre volume_hu_crudo sin
    tocar, tal como exigio el profesor (el umbral en HU se aplica antes de
    normalizar la imagen que entra al modelo).

    aplicar_clahe=False deshabilita el paso para comparaciones/debug; todo
    el pipeline de entrenamiento e inferencia debe dejarlo en True (default)
    para que ambos usen exactamente el mismo preprocesamiento.
    """
    volume = clip_hu(volume, clip_min, clip_max)
    low, high = center - width / 2, center + width / 2
    windowed = np.clip(volume, low, high)
    windowed = (windowed - low) / (high - low)
    windowed = windowed.astype(np.float32)

    if aplicar_clahe and windowed.max() > windowed.min():
        windowed = _sk_exposure.equalize_adapthist(
            windowed, clip_limit=CLAHE_CLIP_LIMIT
        ).astype(np.float32)

    return windowed


def bone_mask(volume_hu_crudo: np.ndarray, threshold: int = BONE_HU_THRESHOLD) -> np.ndarray:
    """
    Mascara gruesa de hueso por umbral de intensidad (preprocesamiento
    clasico, no interviene el modelo). Por si sola incluye tambien
    estructuras metalicas externas al paciente (rieles de la camilla,
    cables), que tienen HU > threshold igual que el hueso. Para descartar
    eso usar clean_bone_mask(), no este umbral solo.
    """
    return volume_hu_crudo > threshold


BODY_HU_THRESHOLD = -300      # separa aire de tejido/hueso del paciente
MIN_NOISE_VOXELS = 3000       # descarta bloques de ruido (streaks/metal-artifact)
                               # mucho mas chicos que la masa osea real; ver
                               # docstring de clean_bone_mask para el porque
                               # de no usar apertura morfologica en su lugar.


def body_mask(volume_hu_crudo: np.ndarray, threshold: int = BODY_HU_THRESHOLD) -> np.ndarray:
    """
    Silueta del cuerpo del paciente: el componente conectado mas grande de
    la mascara HU > threshold, con huecos internos rellenos (pulmones,
    gas intestinal). El cuerpo es, por definicion, una unica masa contigua
    grande; la camilla, rieles y cables externos quedan separados por aire
    y por eso no forman parte de este componente aunque tengan HU alto.

    Relleno en dos pasadas: 2D por corte axial primero (tecnica estandar
    para mascara de cuerpo en CT, cierra cavidades de aire dentro de un
    mismo corte) y luego un relleno 3D final como red de seguridad.
    """
    raw = volume_hu_crudo > threshold

    filled = np.zeros_like(raw)
    for z in range(raw.shape[0]):
        filled[z] = ndi.binary_fill_holes(raw[z])

    labels, n_components = ndi.label(filled, structure=np.ones((3, 3, 3)))
    if n_components == 0:
        return filled

    sizes = ndi.sum(filled, labels, index=np.arange(1, n_components + 1))
    biggest = int(np.argmax(sizes)) + 1
    mask = labels == biggest

    return ndi.binary_fill_holes(mask)


def clean_bone_mask(volume_hu_crudo: np.ndarray, bone_threshold: int = BONE_HU_THRESHOLD,
                     body_threshold: int = BODY_HU_THRESHOLD,
                     min_noise_voxels: int = MIN_NOISE_VOXELS,
                     opening_iterations: int = 0) -> np.ndarray:
    """
    Mascara de hueso limpia, sin camilla ni cables: interseccion entre
    bone_mask() y body_mask(), seguida de un filtro de componentes
    conectados pequenos (rayas de beam hardening y motas de metal-artifact
    que, dentro del cuerpo, forman bloques mucho mas chicos que la masa
    osea completa).

    opening_iterations=0 por defecto A PROPOSITO. Se probo con apertura
    morfologica y destruyo hueso cortical real: a este spacing (~0.7-0.8
    mm/voxel) la cortical solo tiene 2-3 voxeles de grosor, practicamente
    el mismo grosor que las rayas de artefacto, asi que cualquier erosion
    3D borra ambas por igual (crestas iliacas perforadas, sacro
    fragmentado). El filtro por tamano de componente es mas seguro porque
    es una decision de todo-o-nada por bloque completo, nunca le quita una
    capa de voxeles a una estructura que sobrevive el filtro.

    Si sigue quedando alguna raya de artefacto que NO se elimina, es porque
    esta conectada directamente al hueso (nace de un tornillo/implante
    dentro del hueso) y en ese caso ningun filtro por tamano o forma la
    va a separar sin tambien afectar el hueso — documentarlo como
    limitacion conocida en el model card, no forzar mas limpieza aqui.

    Usada por el visualizador 1 (reconstruccion 3D del volumen crudo) y
    reutilizable por el visualizador 3 (reconstruccion final), para no
    duplicar esta logica de limpieza.
    """
    hueso = bone_mask(volume_hu_crudo, bone_threshold)
    cuerpo = body_mask(volume_hu_crudo, body_threshold)
    limpio = hueso & cuerpo

    if opening_iterations > 0:
        limpio = ndi.binary_opening(
            limpio, structure=np.ones((3, 3, 3)), iterations=opening_iterations
        )

    labels, n_components = ndi.label(limpio, structure=np.ones((3, 3, 3)))
    if n_components == 0:
        return limpio
    sizes = ndi.sum(limpio, labels, index=np.arange(1, n_components + 1))

    keep = np.zeros_like(limpio)
    for comp_id, size in enumerate(sizes, start=1):
        if size >= min_noise_voxels:
            keep |= (labels == comp_id)

    return keep


def resize_slice(slice_2d: np.ndarray, size: int = 256, order: int = 1) -> np.ndarray:
    """
    Redimensiona un corte 2D (y, x) a (size, size). order=1 (bilineal) para
    imagenes en ventana [0,1]; usar order=0 (vecino mas cercano) para
    mascaras de etiquetas, pasando is_label=True para conservar valores
    enteros exactos.
    """
    return _sk_resize(
        slice_2d, (size, size),
        order=order, preserve_range=True, anti_aliasing=(order != 0),
    ).astype(slice_2d.dtype if order == 0 else np.float32)


def resize_label_slice(slice_2d: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize de una mascara de fragmentos: vecino mas cercano, sin promediar etiquetas."""
    out = _sk_resize(
        slice_2d, (size, size),
        order=0, preserve_range=True, anti_aliasing=False,
    )
    return np.round(out).astype(slice_2d.dtype)


# ---------------------------------------------------------------------------
# PRUEBA DE HUMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    case_id = sys.argv[1] if len(sys.argv) > 1 else "001"

    caso = load_case(case_id)
    print(f"Caso {caso.case_id}: image {caso.image.shape}, "
          f"label {None if caso.label is None else caso.label.shape}, "
          f"spacing_xyz={caso.spacing_xyz}")

    corte_central = caso.image[caso.image.shape[0] // 2]
    ventaneado = window_hu(corte_central)
    print(f"Corte central ventaneado: min={ventaneado.min():.3f}, "
          f"max={ventaneado.max():.3f}, dtype={ventaneado.dtype}")

    corte_256 = resize_slice(ventaneado, 256)
    print(f"Resize a 256: shape={corte_256.shape}")

    if caso.label is not None:
        label_256 = resize_label_slice(caso.label[caso.image.shape[0] // 2], 256)
        print(f"Resize label a 256: shape={label_256.shape}, "
              f"valores unicos={sorted(np.unique(label_256).tolist())}")

    mascara = bone_mask(caso.image)
    print(f"Bone mask: {mascara.sum()} voxeles de hueso de {mascara.size} totales")

    print("\nOK: pengwin_io.py funciona correctamente.")