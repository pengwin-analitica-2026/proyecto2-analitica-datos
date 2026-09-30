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
    corte = window_hu(caso.image[caso.image.shape[0] // 2])
    corte_256 = resize_slice(corte, 256)
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.transform import resize as _sk_resize
from skimage import exposure as _sk_exposure
from skimage.morphology import ball as _sk_ball

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
WINDOW_CENTER, WINDOW_WIDTH = 400, 1800  # ventana osea -500 a 1300HU
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

# Filtro por profundidad (ver clean_bone_mask). Valores elegidos con el
# diagnostico sobre 20 casos: las islas externas del caso 001 estaban a 1.6 mm
# de la piel; todo componente pequeno real (fragmentos y huesos sin etiquetar)
# estaba a 36 mm o mas. 10 mm cae dentro de ese hueco, sesgado a conservar hueso.
MIN_SURFACE_DEPTH_MM = 10.0   # profundidad minima (mm bajo la piel) de un componente pequeno
BIG_COMPONENT_VOXELS = 50_000  # componentes >= a esto (esqueleto, femures) nunca se filtran
DEPTH_DOWNSAMPLE = 2           # submuestreo del mapa de profundidad (memoria); error ~1-2 mm

# Recuperacion de hueso de baja densidad (solo para la mascara visual del V1).
# Con umbral fijo 250 HU la mascara captura en mediana solo 57% del hueso
# etiquetado (22 casos, diag_umbral.py). Crecimiento por histeresis 150/250 +
# cierre + relleno de cavidades sube la mediana a ~92% dentro de la caja del
# hueso (diag_relleno.py). Se aplica DESPUES de la limpieza y solo a >= 10 mm de
# la piel: aplicado antes, fusiono con el hueso los arcos pegados a la piel del
# caso 001 y el filtro de profundidad dejo de poder borrarlos.
BONE_LOW_HU_THRESHOLD = 150   # umbral bajo del crecimiento (semilla = mascara ya limpia)
CLOSING_RADIUS_VOX = 2        # radio (voxeles) del cierre morfologico que sella grietas


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
                     opening_iterations: int = 0,
                     spacing_zyx: tuple | None = None,
                     min_surface_depth_mm: float = MIN_SURFACE_DEPTH_MM,
                     big_component_voxels: int = BIG_COMPONENT_VOXELS,
                     low_threshold: int | None = BONE_LOW_HU_THRESHOLD,
                     fill_cavities: bool = True,
                     closing_radius: int = CLOSING_RADIUS_VOX) -> np.ndarray:
    """
    Mascara de hueso limpia, sin camilla ni cables: interseccion entre
    bone_mask() y body_mask(), seguida de dos filtros de componentes
    conectados (todo-o-nada por bloque):

    1. Tamano: descarta bloques de menos de min_noise_voxels (rayas de beam
       hardening y motas de metal-artifact dentro del cuerpo).
    2. Profundidad (solo si se pasa spacing_zyx): descarta componentes
       pequenos (< big_component_voxels) que tocan la zona a menos de
       min_surface_depth_mm de la piel. Un fragmento oseo real queda
       rodeado de tejido blando; un cinturon, electrodo o cable que roza la
       silueta del cuerpo queda pegado a la superficie. Los componentes
       grandes (esqueleto, femures) se conservan siempre. Se eligio profundidad
       y no distancia al hueso principal porque un fragmento desplazado puede
       quedar a varios cm del hueso, y esos son justo los que hay que medir.

    spacing_zyx = (sz, sy, sx) en mm, en el orden del volumen (z, y, x). Si
    es None el filtro de profundidad se omite (comportamiento anterior).

    3. Recuperacion de hueso de baja densidad (requiere spacing_zyx): partiendo
    de la mascara ya limpia (filtros 1 y 2), crece hacia voxeles entre
    low_threshold (150 HU) y bone_threshold, solo si estan conectados a ese
    hueso y a >= min_surface_depth_mm de la piel; luego un cierre morfologico
    sella grietas de la cortical y se rellenan las cavidades 3D encerradas
    (interior esponjoso), tambien solo en la zona profunda. Como el crecimiento
    parte de la mascara limpia y no toca la zona cercana a la piel, no puede
    fusionar cinturones o cables con el hueso. low_threshold=None y
    fill_cavities=False reproducen el comportamiento de umbral fijo.
    SOLO para visualizacion: el cierre puede unir huesos separados por un
    espacio fino o tapar una linea de fractura, asi que esta mascara NO debe
    usarse para medir distancias.

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

    # Filtro 1: tamano minimo
    keep_ids = sizes >= min_noise_voxels

    # Mapa de profundidad bajo la piel (submuestreado), si hay spacing
    f = DEPTH_DOWNSAMPLE
    depth = None
    if spacing_zyx is not None:
        depth = ndi.distance_transform_edt(
            cuerpo[::f, ::f, ::f],
            sampling=tuple(sp * f for sp in spacing_zyx),
        )

        # Filtro 2: profundidad, solo para componentes pequenos
        small_ids = np.flatnonzero(keep_ids & (sizes < big_component_voxels)) + 1
        if len(small_ids) > 0:
            objs = ndi.find_objects(labels)
            for k in small_ids:
                sl = objs[k - 1]
                zz, yy, xx = np.nonzero(labels[sl] == k)
                d = depth[(zz + sl[0].start) // f,
                          (yy + sl[1].start) // f,
                          (xx + sl[2].start) // f]
                if d.min() < min_surface_depth_mm:
                    keep_ids[k - 1] = False

    lut = np.zeros(n_components + 1, dtype=bool)
    lut[1:] = keep_ids
    seguro = lut[labels]

    # Filtro 3: recuperacion de hueso tenue, DESPUES de limpiar y solo en profundidad
    if (depth is None or low_threshold is None or low_threshold >= bone_threshold
            or not seguro.any()):
        return seguro

    shape = seguro.shape
    interior = np.repeat(np.repeat(np.repeat(depth >= min_surface_depth_mm, f, axis=0),
                                   f, axis=1), f, axis=2)[:shape[0], :shape[1], :shape[2]]

    bb = ndi.find_objects(cuerpo.astype(np.uint8))[0]
    seg_c = seguro[bb]
    int_c = interior[bb] & cuerpo[bb]
    debil = seg_c | ((volume_hu_crudo[bb] > low_threshold) & int_c)
    lab_w, n_w = ndi.label(debil)                       # conectividad de 6 vecinos
    con_semilla = np.unique(lab_w[seg_c])
    con_semilla = con_semilla[con_semilla > 0]
    lut_w = np.zeros(n_w + 1, dtype=bool)
    lut_w[con_semilla] = True
    crecido = lut_w[lab_w]

    if fill_cavities and closing_radius > 0:
        # Cierre + relleno solo sobre la caja del hueso; pad 'edge' para que la
        # erosion del cierre no se coma hueso que toca el borde del volumen.
        r = closing_radius
        sb = ndi.find_objects(crecido.astype(np.uint8))[0]
        sl2 = tuple(slice(max(b.start - r - 1, 0), b.stop + r + 1) for b in sb)
        sub = np.pad(crecido[sl2], r, mode="edge")
        sub = ndi.binary_closing(sub, structure=_sk_ball(r))
        sub = ndi.binary_fill_holes(sub)[r:-r, r:-r, r:-r]
        crecido[sl2] |= sub & int_c[sl2]

    salida = seguro.copy()
    salida[bb] |= crecido
    return salida


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