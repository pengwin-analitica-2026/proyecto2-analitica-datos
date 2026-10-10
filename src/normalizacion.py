"""Normalización de la orientación de un volumen CT antes del modelo.

Sigue la metodología de Ferro Sánchez et al. (MethodsX 13:103073, 2024), adaptada a la pelvis:

1. **Orientación con los metadatos** (``a_orientacion_estandar``). La cabecera trae la matriz de
   dirección: una columna por eje de la imagen (x, y, z de los índices), expresada en el espacio
   del paciente. Con productos punto se verifica que los ejes sean perpendiculares y se decide con
   qué eje del paciente se alinea cada uno; con el producto cruz se obtiene la normal a los cortes y
   se detecta si el volumen está espejado. Luego se permutan y voltean los ejes (o se remuestrea, si
   la adquisición es oblicua) hasta que la dirección sea la identidad, la orientación de 66 de los
   100 casos de PENGWIN. Funciona igual con ``.mha`` y NIfTI, porque SimpleITK lee las dos
   cabeceras con la misma interfaz. En los 100 casos de PENGWIN da exactamente lo mismo que
   ``orientacion.to_canonical`` (ver ``tests/test_normalizacion.py``).

2. **Giro axial del paciente** (``estimar_giro_axial`` y ``corregir_giro_axial``). La inclinación
   del paciente dentro del escáner no está en los metadatos. El artículo la mide con la línea entre
   dos estructuras simétricas; aquí se usa la simetría izquierda-derecha de toda la pelvis en una
   proyección de máxima intensidad (MIP) axial del hueso: el ángulo correcto es el que hace que la
   silueta coincida mejor con su espejo. Usar la silueta completa, y no dos puntos, la hace menos
   sensible a un fragmento desplazado. Solo se corrige si el ángulo es claro (``giro_minimo`` y
   ``mejora_minima``), para no "enderezar" pacientes que ya estaban bien.

Todo usa solo la imagen y su cabecera, nunca las etiquetas, así que vale igual en entrenamiento,
evaluación y la demo. La etiqueta, si se pasa, recibe exactamente la misma transformación (con
vecino más cercano, para no inventar IDs de fragmento).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

TOLERANCIA = 1e-3


# ---------------------------------------------------------------------------
# Paso 1: orientación estándar con los metadatos
# ---------------------------------------------------------------------------
@dataclass
class DiagnosticoOrientacion:
    """Lo que dice la cabecera, calculado con productos punto y cruz."""
    direccion: np.ndarray            # 3x3, columna c = eje c de los índices (x, y, z) en el paciente
    ortogonal: bool                  # filas · columnas ≈ 0 (y los demás pares)
    espejado: bool                   # (filas × columnas) · cortes < 0
    oblicuo: bool                    # algún eje no coincide con un eje del paciente
    ejes_paciente: tuple[int, int, int] = field(default=(0, 1, 2))  # eje del paciente de cada eje
    signos: tuple[int, int, int] = field(default=(1, 1, 1))         # +1 / -1 por eje

    @property
    def es_estandar(self) -> bool:
        return (not self.oblicuo and self.ejes_paciente == (0, 1, 2)
                and self.signos == (1, 1, 1))


def diagnosticar(direccion) -> DiagnosticoOrientacion:
    """Analiza la matriz de dirección de la cabecera con productos punto y cruz."""
    d = np.asarray(direccion, dtype=np.float64).reshape(3, 3)
    filas, columnas, cortes = d[:, 0], d[:, 1], d[:, 2]

    # Producto punto: los tres ejes deben ser perpendiculares entre sí.
    ortogonal = (abs(np.dot(filas, columnas)) < TOLERANCIA
                 and abs(np.dot(filas, cortes)) < TOLERANCIA
                 and abs(np.dot(columnas, cortes)) < TOLERANCIA)

    # Producto cruz: normal a los cortes. Si apunta al revés del tercer eje, está espejado.
    normal = np.cross(filas, columnas)
    espejado = bool(np.dot(normal, cortes) < 0)

    # Producto punto con cada eje del paciente (e_x, e_y, e_z) = la fila de la matriz: el mayor
    # en valor absoluto dice con qué eje del paciente se alinea cada eje de la imagen.
    proyecciones = np.abs(d)                       # |e_p · eje_c| para cada p (fila) y c (columna)
    ejes = tuple(int(np.argmax(proyecciones[:, c])) for c in range(3))
    signos = tuple(int(np.sign(d[ejes[c], c])) or 1 for c in range(3))
    oblicuo = (len(set(ejes)) < 3
               or any(abs(proyecciones[ejes[c], c] - 1.0) > TOLERANCIA for c in range(3)))
    return DiagnosticoOrientacion(d, bool(ortogonal), espejado, bool(oblicuo), ejes, signos)


def a_orientacion_estandar(imagen: sitk.Image, etiqueta: sitk.Image | None = None,
                           ) -> tuple[sitk.Image, sitk.Image | None, DiagnosticoOrientacion]:
    """Lleva el volumen (y su etiqueta) a dirección identidad usando solo la cabecera.

    Sin oblicuidad, solo permuta y voltea ejes: no interpola, así que la imagen no pierde calidad.
    Con oblicuidad, remuestrea a una grilla con dirección identidad (lineal para la imagen, vecino
    más cercano para la etiqueta).
    """
    diag = diagnosticar(imagen.GetDirection())
    if not diag.ortogonal:
        raise ValueError("La matriz de dirección no es ortogonal: la cabecera está dañada.")
    if diag.es_estandar:
        return imagen, etiqueta, diag
    if not diag.oblicuo:
        return (_permutar_y_voltear(imagen, diag),
                _permutar_y_voltear(etiqueta, diag) if etiqueta is not None else None, diag)
    return (_remuestrear_identidad(imagen, sitk.sitkLinear),
            _remuestrear_identidad(etiqueta, sitk.sitkNearestNeighbor, referencia=imagen)
            if etiqueta is not None else None, diag)


def _permutar_y_voltear(img: sitk.Image, diag: DiagnosticoOrientacion) -> sitk.Image:
    """Permuta y voltea ejes según el diagnóstico, sin interpolar.

    Dirección identidad en ITK equivale al código de orientación «LPS» de DICOM (cada eje de índice
    crece hacia Left, Posterior, Superior en el sistema del paciente de ITK). DICOMOrient hace la
    permutación y los volteos que el diagnóstico calculó con los productos punto, y conserva el
    origen físico correcto.
    """
    salida = sitk.DICOMOrient(img, "LPS")
    restante = diagnosticar(salida.GetDirection())
    if not restante.es_estandar:   # no debería pasar si la entrada no es oblicua
        raise RuntimeError(f"No se pudo llevar a dirección identidad: {restante}")
    return salida


def _remuestrear_identidad(img: sitk.Image, interpolador, referencia: sitk.Image | None = None
                           ) -> sitk.Image:
    ref = referencia if referencia is not None else img
    d = np.array(ref.GetDirection()).reshape(3, 3)
    tam = np.array(ref.GetSize())
    esp = np.array(ref.GetSpacing())
    origen = np.array(ref.GetOrigin())
    # Esquinas del volumen en el espacio físico → caja que las contiene en la grilla nueva.
    esquinas = np.array([[i, j, k] for i in (0, tam[0] - 1) for j in (0, tam[1] - 1)
                         for k in (0, tam[2] - 1)], dtype=np.float64)
    fisicas = origen + (esquinas * esp) @ d.T
    paso = np.full(3, esp.min())
    nuevo_origen = fisicas.min(axis=0)
    nuevo_tam = np.ceil((fisicas.max(axis=0) - nuevo_origen) / paso).astype(int) + 1
    relleno = -1024 if interpolador == sitk.sitkLinear else 0
    return sitk.Resample(img, [int(v) for v in nuevo_tam], sitk.Transform(), interpolador,
                         tuple(nuevo_origen), tuple(paso), tuple(np.eye(3).ravel()),
                         relleno, img.GetPixelID())


def leer_normalizado(ruta_imagen: str, ruta_etiqueta: str | None = None):
    """Lee un .mha o NIfTI y lo devuelve en orientación estándar como arreglos (z, y, x).

    Devuelve (imagen, etiqueta o None, spacing_xyz, diagnóstico).
    """
    img = sitk.ReadImage(str(ruta_imagen))
    lbl = sitk.ReadImage(str(ruta_etiqueta)) if ruta_etiqueta else None
    img, lbl, diag = a_orientacion_estandar(img, lbl)
    return (sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(lbl) if lbl is not None else None,
            img.GetSpacing(), diag)


# ---------------------------------------------------------------------------
# Paso 2: giro axial del paciente (no está en los metadatos)
# ---------------------------------------------------------------------------
@dataclass
class EstimacionGiro:
    angulo: float          # grados; girar el volumen en -angulo lo endereza
    simetria: float        # Dice entre la silueta enderezada y su espejo (1 = simétrica)
    simetria_sin_girar: float
    corregir: bool         # True si el ángulo es claro y vale la pena corregir


def mip_axial_hueso(volumen_hu: np.ndarray, spacing_xyz, umbral_hu: float = 250,
                    mm_por_pixel: float = 1.0) -> np.ndarray:
    """Silueta del hueso vista desde arriba (MIP axial), remuestreada a píxeles cuadrados."""
    silueta = (volumen_hu > umbral_hu).any(axis=0).astype(np.float32)
    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    silueta = ndi.zoom(silueta, (sy / mm_por_pixel, sx / mm_por_pixel), order=0)
    # Se queda con lo más grande (la pelvis), sin camilla ni objetos sueltos.
    etiquetas, n = ndi.label(silueta > 0.5)
    if n > 1:
        tamanos = ndi.sum(silueta > 0.5, etiquetas, index=range(1, n + 1))
        silueta = np.isin(etiquetas, np.argsort(tamanos)[-3:] + 1).astype(np.float32)
    return silueta


def _mas_ancha_que_profunda(mascara: np.ndarray) -> bool:
    """La pelvis vista desde arriba es más ancha (izquierda-derecha) que profunda (adelante-atrás)."""
    ys, xs = np.nonzero(mascara)
    return bool(xs.var() >= ys.var())


def _simetria(silueta: np.ndarray, angulo: float) -> float:
    """Dice entre la silueta girada y su espejo izquierda-derecha.

    Restricción anatómica: si al girar la pelvis queda más profunda que ancha, el ángulo no es
    válido (penalización de 1). Sin ella, en pelvis casi simétricas también de adelante hacia
    atrás el método confunde el ángulo correcto con uno a 90° (visto en 044, 067, 078 y 099).
    """
    girada = ndi.rotate(silueta, angulo, reshape=False, order=1) > 0.5
    if not girada.any():
        return 0.0
    penalizacion = 0.0 if _mas_ancha_que_profunda(girada) else 1.0
    # Centra la silueta en x para que el espejo sea alrededor de su propio eje.
    cx = ndi.center_of_mass(girada)[1]
    girada = ndi.shift(girada.astype(np.float32), (0, girada.shape[1] / 2 - 0.5 - cx), order=0) > 0.5
    espejo = girada[:, ::-1]
    inter = np.logical_and(girada, espejo).sum()
    return float(2 * inter / (girada.sum() + espejo.sum())) - penalizacion


def estimar_giro_axial(silueta: np.ndarray, rango: float = 89.0, paso: float = 1.0,
                       giro_minimo: float = 5.0, mejora_minima: float = 0.02) -> EstimacionGiro:
    """Ángulo axial que hace a la silueta más simétrica izquierda-derecha.

    Búsqueda gruesa cada ``paso`` grados en ±``rango`` y luego fina cada 0.25°. ``corregir`` solo
    es True si el giro supera ``giro_minimo`` y la simetría mejora al menos ``mejora_minima``
    respecto a no girar: así un paciente bien acostado se deja como está.
    """
    angulos = np.arange(-rango, rango + paso / 2, paso)
    puntajes = [_simetria(silueta, a) for a in angulos]
    mejor = float(angulos[int(np.argmax(puntajes))])
    finos = np.arange(mejor - paso, mejor + paso + 0.125, 0.25)
    puntajes_finos = [_simetria(silueta, a) for a in finos]
    mejor = float(finos[int(np.argmax(puntajes_finos))])
    s_mejor, s_cero = max(puntajes_finos), max(_simetria(silueta, 0.0), 0.0)
    corregir = abs(mejor) >= giro_minimo and (s_mejor - s_cero) >= mejora_minima
    # rotate(+a) endereza la silueta; el paciente estaba girado -a.
    return EstimacionGiro(angulo=-mejor, simetria=s_mejor, simetria_sin_girar=s_cero,
                          corregir=bool(corregir))


def corregir_giro_axial(volumen: np.ndarray, angulo: float, es_etiqueta: bool = False,
                        relleno: float | None = None, mm_xy: tuple[float, float] | None = None
                        ) -> np.ndarray:
    """Gira cada corte axial en -``angulo`` grados (endereza). Etiqueta: vecino más cercano.

    ``mm_xy`` = mm por píxel (x, y). Si se da, el giro se hace en milímetros reales:

        entrada = D⁻¹ · R · D · (salida − centro) + centro,     D = diag(mm_y, mm_x)

    así una pelvis con píxeles no cuadrados (caso del caché, que estira cada corte a S×S) gira
    sin deformarse. Con píxeles cuadrados da lo mismo que ``ndi.rotate`` (ver las pruebas).
    """
    if angulo == 0:
        return volumen
    if relleno is None:
        relleno = 0 if es_etiqueta else float(volumen.min())
    orden = 0 if es_etiqueta else 1
    if mm_xy is None:
        return ndi.rotate(volumen, -angulo, axes=(1, 2), reshape=False, order=orden,
                          mode="constant", cval=relleno).astype(volumen.dtype)
    a = np.radians(-angulo)
    # Convención de ndi.rotate en coordenadas (fila, columna) = (y, x).
    r = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    d = np.diag([float(mm_xy[1]), float(mm_xy[0])])
    m = np.eye(3)
    m[1:, 1:] = np.linalg.inv(d) @ r @ d
    centro = (np.array(volumen.shape, dtype=float) - 1) / 2
    return ndi.affine_transform(volumen, m, offset=centro - m @ centro, order=orden,
                                mode="constant", cval=relleno).astype(volumen.dtype)


# ---------------------------------------------------------------------------
# Paso 2 sobre el caché del modelo (imágenes uint8 ya ventaneadas)
# ---------------------------------------------------------------------------
# El caché guarda cada corte como uint8 = 255 · ventana(HU) ** 1.5, con ventana C400/W1800
# (−500 a 1300 HU). El umbral óseo de 250 HU cae en 255 · (750/1800) ** 1.5 ≈ 68.6.
VENTANA_MIN_HU, VENTANA_ANCHO_HU, GAMMA_CACHE = -500.0, 1800.0, 1.5


def umbral_u8(hu: float = 250.0) -> float:
    """Valor uint8 del caché que corresponde a ``hu`` HU (después de ventana y gamma)."""
    v = float(np.clip((hu - VENTANA_MIN_HU) / VENTANA_ANCHO_HU, 0.0, 1.0))
    return 255.0 * v ** GAMMA_CACHE


def silueta_de_cache(imagenes_u8: np.ndarray, mm_xy: tuple[float, float],
                     umbral_hu: float = 250.0) -> np.ndarray:
    """MIP axial de hueso a partir del caché, remuestreada a 1 mm por píxel (píxeles cuadrados)."""
    silueta = (imagenes_u8 > umbral_u8(umbral_hu)).any(axis=0).astype(np.float32)
    silueta = ndi.zoom(silueta, (float(mm_xy[1]), float(mm_xy[0])), order=0)
    etiquetas, n = ndi.label(silueta > 0.5)
    if n > 1:
        tamanos = ndi.sum(silueta > 0.5, etiquetas, index=range(1, n + 1))
        silueta = np.isin(etiquetas, np.argsort(tamanos)[-3:] + 1).astype(np.float32)
    return silueta


def enderezar_caso(imagenes_u8: np.ndarray, etiquetas: np.ndarray | None,
                   mm_xy: tuple[float, float], **kwargs
                   ) -> tuple[np.ndarray, np.ndarray | None, EstimacionGiro]:
    """Paso 2 sobre un caso del caché (que ya viene en orientación estándar por el paso 1).

    Estima el giro con la silueta en milímetros y, solo si ``corregir``, gira la imagen
    (bilineal, esquinas en 0) y la etiqueta (vecino más cercano) en milímetros reales.
    Usa solo la imagen: la etiqueta no interviene en la estimación.
    """
    est = estimar_giro_axial(silueta_de_cache(imagenes_u8, mm_xy), **kwargs)
    if not est.corregir:
        return imagenes_u8, etiquetas, est
    imagenes = corregir_giro_axial(imagenes_u8, est.angulo, relleno=0, mm_xy=mm_xy)
    if etiquetas is not None:
        etiquetas = corregir_giro_axial(etiquetas, est.angulo, es_etiqueta=True, mm_xy=mm_xy)
    return imagenes, etiquetas, est


def normalizar(volumen_hu: np.ndarray, spacing_xyz, etiqueta: np.ndarray | None = None,
               **kwargs) -> tuple[np.ndarray, np.ndarray | None, EstimacionGiro]:
    """Paso 2 completo sobre un volumen ya en orientación estándar (paso 1)."""
    est = estimar_giro_axial(mip_axial_hueso(volumen_hu, spacing_xyz), **kwargs)
    if not est.corregir:
        return volumen_hu, etiqueta, est
    return (corregir_giro_axial(volumen_hu, est.angulo),
            corregir_giro_axial(etiqueta, est.angulo, es_etiqueta=True)
            if etiqueta is not None else None, est)
