"""Entrada de un CT propio para la demo: leer, orientar y preprocesar, sin torch.

El dashboard deja subir un volumen que no es de PENGWIN, en dos formas:

- **Un archivo volumétrico**: ``.mha``, ``.mhd`` con su ``.raw`` o ``.zraw``, ``.nii``, ``.nii.gz`` o
  ``.nrrd``. SimpleITK lee la cabecera de todos con la misma interfaz.
- **Una carpeta de cortes DICOM** (un archivo por corte). Se arma la serie con GDCM, que ordena los
  cortes por su posición en el paciente y aplica ``RescaleSlope`` y ``RescaleIntercept``, así la
  imagen queda en HU. Si la carpeta trae varias series, se usa la que tiene más cortes.

Después se lleva el volumen a la orientación estándar con los metadatos
(``normalizacion.a_orientacion_estandar``): es la misma orientación canónica con la que se
entrenaron H6, H7 y H8, y en los 100 casos de PENGWIN da lo mismo que ``orientacion.to_canonical``.
Una carpeta de PNG o JPG no sirve: esas imágenes no guardan HU ni el espaciado en milímetros, y el
modelo y la distancia los necesitan.

Este módulo no importa torch, así que también corre en Streamlit Community Cloud, donde la inferencia
la hace la API de AWS (``inferencia_remota.py``).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from build_case_cache import to_uint8
from pengwin_io import GAMMA, resize_slice, window_hu

EXTENSIONES_VOLUMEN = (".mha", ".mhd", ".nii", ".nii.gz", ".nrrd", ".nhdr")
EXTENSIONES_COMPANERAS = (".raw", ".zraw", ".gz")          # datos de un .mhd o .nhdr
EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
MIN_CORTES = 8


@dataclass
class VolumenSubido:
    nombre: str                     # identificador corto para la carpeta y el dashboard
    image: np.ndarray               # (z, y, x) en HU, orientación estándar
    spacing_xyz: tuple[float, float, float]
    fuente: str                     # "archivo .mha", "serie DICOM (312 cortes)", ...
    orientacion: dict               # qué hizo la orientación con los metadatos
    pasos: object = None            # pasos_preproceso.Pasos con las etapas «crudo» y «orientado»


def preprocess_volume(image_hu: np.ndarray, size: int = 384, gamma: float = GAMMA) -> np.ndarray:
    """Volumen HU crudo [z,y,x] → uint8 [z,size,size], igual que ``build_case_cache`` (modo stretch)."""
    out = np.empty((len(image_hu), size, size), dtype=np.uint8)
    for z in range(len(image_hu)):
        out[z] = to_uint8(resize_slice(window_hu(image_hu[z], gamma=gamma), size))
    return out


def cache_spacing_xyz(spacing_xyz: tuple[float, float, float], shape_zyx: tuple[int, int, int],
                      size: int) -> tuple[float, float, float]:
    """Spacing (x,y,z) en mm del volumen estirado a size×size por corte.

    Cada píxel del caché cubre más o menos milímetros que el original según la forma del corte.
    Misma fórmula que ``volume_postprocess.cache_spacing_xyz``, pero desde el header.
    """
    sx, sy, sz = (float(v) for v in spacing_xyz)
    return (sx * shape_zyx[2] / size, sy * shape_zyx[1] / size, sz)


def huella(archivos: list[tuple[str, bytes]]) -> str:
    """Hash corto del contenido subido: la misma subida no se procesa dos veces."""
    h = hashlib.sha1()
    for nombre, datos in sorted(archivos, key=lambda a: a[0]):
        h.update(Path(nombre).name.encode())
        h.update(datos)
    return h.hexdigest()[:10]


def _extension(nombre: str) -> str:
    nombre = nombre.lower()
    return ".nii.gz" if nombre.endswith(".nii.gz") else Path(nombre).suffix


def _nombre_corto(nombre: str) -> str:
    base = Path(nombre).name
    for ext in sorted(EXTENSIONES_VOLUMEN, key=len, reverse=True):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    limpio = "".join(c if c.isalnum() or c in "-_" else "_" for c in base).strip("_")
    return (limpio or "volumen")[:40]


def leer_subida(archivos: list[tuple[str, bytes]], orientar: bool = True) -> VolumenSubido:
    """Lee lo que subió el usuario: un volumen o una carpeta DICOM. ``archivos`` = [(nombre, bytes)].

    Con ``orientar=True`` (modelos con orientación canónica) lleva el volumen a la orientación estándar
    con la matriz de dirección de la cabecera.
    """
    import SimpleITK as sitk
    if not archivos:
        raise ValueError("No se subió ningún archivo.")
    archivos = [(n, d) for n, d in archivos if not Path(n).name.startswith(".")]   # .DS_Store y similares
    volumenes = [a for a in archivos if _extension(a[0]) in EXTENSIONES_VOLUMEN]
    if any(_extension(n) in EXTENSIONES_IMAGEN for n, _ in archivos) and not volumenes:
        raise ValueError("La carpeta trae imágenes PNG/JPG. Esas imágenes no guardan las unidades "
                         "Hounsfield ni el tamaño del vóxel en mm, que el modelo y la distancia "
                         "necesitan. Sube el volumen (.mha, .nii, .nrrd) o la serie DICOM original.")
    with tempfile.TemporaryDirectory() as carpeta:
        carpeta = Path(carpeta)
        for nombre, datos in archivos:
            destino = carpeta / Path(nombre).name
            destino.write_bytes(datos)
        if volumenes:
            if len(volumenes) > 1:
                raise ValueError("Sube un solo volumen a la vez (llegaron "
                                 f"{', '.join(Path(n).name for n, _ in volumenes)}).")
            nombre = Path(volumenes[0][0]).name
            try:
                imagen = sitk.ReadImage(str(carpeta / nombre))
            except RuntimeError as error:
                raise ValueError(f"SimpleITK no pudo leer {nombre}: {str(error).splitlines()[0][:200]}. "
                                 "Si es un .mhd o .nhdr, sube también su archivo de datos.") from error
            fuente, corto = f"archivo {_extension(nombre)}", _nombre_corto(nombre)
        else:
            imagen, n_cortes = _leer_dicom(carpeta)
            primero = Path(archivos[0][0])
            carpeta_origen = primero.parent.name if str(primero.parent) not in ("", ".") else ""
            fuente, corto = f"serie DICOM ({n_cortes} cortes)", _nombre_corto(carpeta_origen or "dicom")
    if imagen.GetDimension() != 3:
        raise ValueError(f"Se esperaba un volumen 3D y llegó uno de {imagen.GetDimension()} dimensiones.")
    if imagen.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("El volumen tiene varios canales por vóxel (¿color?); un CT tiene uno.")
    if imagen.GetSize()[2] < MIN_CORTES:
        raise ValueError(f"El volumen tiene {imagen.GetSize()[2]} cortes; se necesitan al menos {MIN_CORTES}.")
    from normalizacion import diagnosticar
    from pasos_preproceso import Pasos, texto_matriz
    matriz = texto_matriz(imagen.GetDirection())
    diag = diagnosticar(imagen.GetDirection())
    orientacion = {"direccion_original": [round(float(v), 4) for v in imagen.GetDirection()],
                   "estandar": True, "accion": "ninguna"}
    pasos = Pasos()
    pasos.agregar_hu("crudo", "Crudo, como viene en el archivo", f"Matriz de dirección de la cabecera: {matriz}. " + (
        "Es la identidad: el volumen ya viene en la orientación estándar." if diag.es_estandar else
        "La adquisición es oblicua: los ejes del volumen no coinciden con los del paciente." if diag.oblicuo else
        "No es la identidad: el volumen está guardado " + ("girado 180° en el plano (LPI)."
        if diag.ejes_paciente == (0, 1, 2) and diag.signos == (-1, -1, 1) else
        "con los ejes permutados o volteados respecto al paciente.")),
        _arreglo(imagen), imagen.GetSpacing(), matriz=matriz)
    if orientar:
        from normalizacion import a_orientacion_estandar
        imagen, _, diag = a_orientacion_estandar(imagen)
        orientacion.update(estandar=diag.es_estandar, oblicuo=diag.oblicuo, espejado=diag.espejado,
                           accion="ninguna (ya estaba en la orientación estándar)" if diag.es_estandar
                           else "remuestreado a dirección identidad (adquisición oblicua)" if diag.oblicuo
                           else "ejes permutados y volteados según la cabecera")
        pasos.agregar_hu("orientado", "Orientado con los metadatos", (
            "El metadato no indica ningún giro: no se cambió nada." if diag.es_estandar else
            "El metadato indica una adquisición oblicua: se remuestreó a la orientación estándar." if diag.oblicuo
            else "El metadato dice que viene girado o con los ejes cambiados: se permutaron y voltearon los "
            "ejes (sin interpolar) hasta la orientación estándar, con el sacro abajo."),
            _arreglo(imagen), imagen.GetSpacing())
    return VolumenSubido(nombre=corto, image=_arreglo(imagen),
                         spacing_xyz=tuple(float(s) for s in imagen.GetSpacing()),
                         fuente=fuente, orientacion=orientacion, pasos=pasos)


def _arreglo(imagen) -> np.ndarray:
    """(z, y, x) en HU como int16 (o float32 si el archivo es de punto flotante)."""
    import SimpleITK as sitk
    image = sitk.GetArrayFromImage(imagen)
    if image.dtype.kind == "f":
        return image.astype(np.float32)
    if image.dtype != np.int16:
        return np.clip(image, -32768, 32767).astype(np.int16)
    return image


def _leer_dicom(carpeta: Path):
    import SimpleITK as sitk
    lector = sitk.ImageSeriesReader()
    series = lector.GetGDCMSeriesIDs(str(carpeta))
    if not series:
        raise ValueError("No encontré un volumen (.mha, .nii, .nrrd) ni una serie DICOM entre los archivos subidos.")
    archivos = max((lector.GetGDCMSeriesFileNames(str(carpeta), s) for s in series), key=len)
    if len(archivos) < MIN_CORTES:
        raise ValueError(f"La serie DICOM tiene {len(archivos)} cortes; se necesitan al menos {MIN_CORTES}.")
    lector.SetFileNames(archivos)
    try:
        return lector.Execute(), len(archivos)
    except RuntimeError as error:
        raise ValueError(f"No pude armar la serie DICOM: {str(error).splitlines()[0][:200]}") from error
