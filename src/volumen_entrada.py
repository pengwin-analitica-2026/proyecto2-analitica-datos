"""Entrada de un CT propio para la demo: leer, orientar y preprocesar, sin torch.

El dashboard deja subir un volumen que no es de PENGWIN, en dos formas:

- **Un archivo volumétrico**: ``.mha``, ``.mhd`` con su ``.raw`` o ``.zraw``, ``.nii``, ``.nii.gz`` o
  ``.nrrd``. SimpleITK lee la cabecera de todos con la misma interfaz.
- **Una carpeta de cortes .mha** (o .nii, .nrrd: un archivo por corte). Se apilan en un volumen
  (``_leer_cortes``): ordenados por su posición en el paciente si cada archivo la trae, o por nombre y
  con el grosor entre cortes que indique el usuario si son imágenes 2D.
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
import re
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


def leer_subida(archivos: list[tuple[str, bytes]], orientar: bool = True,
                grosor_mm: float | None = None) -> VolumenSubido:
    """Lee lo que subió el usuario: un volumen, una carpeta DICOM o una carpeta de cortes (un .mha por
    corte). ``archivos`` = [(nombre, bytes)].

    ``grosor_mm`` es la distancia entre cortes para una carpeta de cortes 2D, que no guardan su posición;
    si los cortes la traen (archivos 3D de un solo corte), se usa la de los archivos. En un volumen o una
    serie DICOM reemplaza la distancia entre cortes de la cabecera, para corregir una cabecera errónea.

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
        primero = Path(archivos[0][0])
        carpeta_origen = primero.parent.name if str(primero.parent) not in ("", ".") else ""
        if len(volumenes) > 1:
            # Una carpeta donde cada archivo (.mha, .nii, ...) es un corte: se apilan en un volumen.
            imagen, n_cortes, orden = _leer_cortes([carpeta / Path(n).name for n, _ in volumenes], grosor_mm)
            ext = _extension(volumenes[0][0])
            fuente = f"carpeta de cortes {ext} ({n_cortes} cortes, ordenados por {orden})"
            corto = _nombre_corto(carpeta_origen or f"cortes{ext}")
        elif volumenes:
            nombre = Path(volumenes[0][0]).name
            try:
                imagen = _leer_compacto(str(carpeta / nombre))
            except RuntimeError as error:
                raise ValueError(f"SimpleITK no pudo leer {nombre}: {str(error).splitlines()[0][:200]}. "
                                 "Si es un .mhd o .nhdr, sube también su archivo de datos.") from error
            fuente, corto = f"archivo {_extension(nombre)}", _nombre_corto(nombre)
        else:
            imagen, n_cortes = _leer_dicom(carpeta)
            fuente, corto = f"serie DICOM ({n_cortes} cortes)", _nombre_corto(carpeta_origen or "dicom")
    if imagen.GetDimension() != 3:
        raise ValueError(f"Se esperaba un volumen 3D y llegó uno de {imagen.GetDimension()} dimensiones.")
    if grosor_mm and grosor_mm > 0 and len(volumenes) <= 1:
        # Volumen o serie DICOM: el usuario corrige el grosor entre cortes de la cabecera (p. ej. un .mha que
        # dice 100 mm; ver calidad_entrada.py). Se cambia el espaciado del eje de cortes antes de orientar.
        espaciado = list(imagen.GetSpacing())
        original, espaciado[2] = espaciado[2], float(grosor_mm)
        imagen.SetSpacing(espaciado)
        fuente += f" · grosor entre cortes corregido a {grosor_mm:g} mm (la cabecera decía {original:g} mm)"
    if imagen.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("El volumen tiene varios canales por vóxel (¿color?); un CT tiene uno.")
    if imagen.GetSize()[2] < MIN_CORTES:
        raise ValueError(f"El volumen tiene {imagen.GetSize()[2]} cortes; se necesitan al menos {MIN_CORTES}.")
    from normalizacion import diagnosticar
    from pasos_preproceso import Pasos, texto_matriz
    # Memoria: el tipo se ajusta dentro de SimpleITK (int16 o float32) y las vistas de los pasos usan
    # GetArrayViewFromImage, sin copiar el volumen. Así un .mha de 230 MB no se multiplica en RAM.
    imagen = _tipo_compacto(imagen)
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
        sitk.GetArrayViewFromImage(imagen), imagen.GetSpacing(), matriz=matriz)
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
            sitk.GetArrayViewFromImage(imagen), imagen.GetSpacing())
    image = sitk.GetArrayFromImage(imagen)
    spacing_xyz = tuple(float(s) for s in imagen.GetSpacing())
    del imagen
    return VolumenSubido(nombre=corto, image=image, spacing_xyz=spacing_xyz,
                         fuente=fuente, orientacion=orientacion, pasos=pasos)


def _clave_natural(ruta: Path) -> list:
    """corte2.mha antes que corte10.mha."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", ruta.name)]


def _leer_cortes(rutas: list[Path], grosor_mm: float | None = None):
    """Apila una carpeta de cortes (un archivo por corte) en un volumen 3D de SimpleITK.

    Si cada archivo es 3D de un solo corte, trae su origen y su dirección: se ordenan por la posición a lo
    largo de la normal del corte y el espaciado en z sale de la distancia entre posiciones, como hace GDCM
    con una serie DICOM. Si son 2D, no hay posición: se ordenan por nombre (orden natural) y hace falta
    ``grosor_mm``, porque el modelo y las distancias en mm necesitan el espaciado real.
    Devuelve (imagen, número de cortes, cómo se ordenaron).
    """
    import SimpleITK as sitk
    cortes = []
    for ruta in rutas:
        try:
            corte = _leer_compacto(str(ruta))
        except RuntimeError as error:
            raise ValueError(f"SimpleITK no pudo leer {ruta.name}: {str(error).splitlines()[0][:200]}") from error
        if corte.GetDimension() == 3 and corte.GetSize()[2] != 1:
            raise ValueError(f"Llegaron varios volúmenes completos ({ruta.name} tiene {corte.GetSize()[2]} cortes). "
                             "Sube un solo volumen, o una carpeta donde cada archivo sea un corte.")
        if corte.GetDimension() not in (2, 3) or corte.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"{ruta.name} no es un corte de CT (un plano en escala de grises).")
        cortes.append((ruta, corte))
    if len(cortes) < MIN_CORTES:
        raise ValueError(f"La carpeta tiene {len(cortes)} cortes; se necesitan al menos {MIN_CORTES}.")
    tamanos = {tuple(c.GetSize()[:2]) for _, c in cortes}
    if len(tamanos) > 1:
        raise ValueError(f"Los cortes no tienen todos el mismo tamaño ({', '.join(f'{a}×{b}' for a, b in sorted(tamanos))}).")
    primero = cortes[0][1]
    sx, sy = (float(v) for v in primero.GetSpacing()[:2])
    posiciones = None
    if primero.GetDimension() == 3 and all(c.GetDimension() == 3 for _, c in cortes):
        direccion = np.array(primero.GetDirection(), dtype=float).reshape(3, 3)
        normal = direccion[:, 2]
        posiciones = np.array([float(np.dot(c.GetOrigin(), normal)) for _, c in cortes])
        if len(np.unique(np.round(posiciones, 3))) < len(posiciones):
            posiciones = None                                 # posiciones repetidas: no sirven para ordenar
    if posiciones is not None:
        orden = np.argsort(posiciones)
        saltos = np.diff(posiciones[orden])
        dz = float(np.median(saltos))
        if saltos.max() > 1.5 * dz + 1e-3:
            raise ValueError(f"Faltan cortes: la distancia entre cortes va de {saltos.min():.2f} a {saltos.max():.2f} mm.")
        origen = cortes[orden[0]][1].GetOrigin()
        como = "posición en el paciente"
    else:
        if not grosor_mm or grosor_mm <= 0:
            raise ValueError("Los cortes son imágenes 2D: no guardan su posición en el paciente, así que no se sabe "
                             "la distancia entre ellos, que el modelo y las distancias en mm necesitan. Indica el "
                             "grosor entre cortes en mm (está en los datos del estudio; suele ser 0.5 a 3 mm).")
        orden = sorted(range(len(cortes)), key=lambda i: _clave_natural(cortes[i][0]))
        dz = float(grosor_mm)
        d2 = np.array(primero.GetDirection(), dtype=float).reshape(primero.GetDimension(), -1)[:2, :2]
        direccion = np.eye(3)
        direccion[:2, :2] = d2
        origen = (*primero.GetOrigin()[:2], 0.0)
        como = f"nombre, {dz:g} mm entre cortes"
    ancho, alto = tamanos.pop()
    volumen = np.stack([sitk.GetArrayFromImage(cortes[i][1]).reshape(alto, ancho) for i in orden])
    imagen = sitk.GetImageFromArray(volumen)
    imagen.SetSpacing((sx, sy, dz))
    imagen.SetOrigin(tuple(float(v) for v in origen))
    imagen.SetDirection(tuple(float(v) for v in direccion.ravel()))
    return imagen, len(cortes), como


def _leer_compacto(ruta: str):
    """Lee el volumen directo en int16 (o float32 si es de punto flotante), sin una copia de 32 bits.

    Muchos .mha guardan los HU con 32 bits por vóxel: leerlos así y convertir después duplicaba la
    memoria (un .mha de 420 MB ocupaba 630 MB solo al leerlo)."""
    import SimpleITK as sitk
    lector = sitk.ImageFileReader()
    lector.SetFileName(ruta)
    lector.ReadImageInformation()
    flotante = lector.GetPixelID() in (sitk.sitkFloat32, sitk.sitkFloat64)
    if lector.GetNumberOfComponents() == 1:
        lector.SetOutputPixelType(sitk.sitkFloat32 if flotante else sitk.sitkInt16)
    return lector.Execute()


def _tipo_compacto(imagen):
    """El volumen en int16 (HU) o float32 si es de punto flotante, convertido dentro de SimpleITK."""
    import SimpleITK as sitk
    tipo = imagen.GetPixelID()
    if tipo in (sitk.sitkInt16,  sitk.sitkFloat32):
        return imagen
    if tipo == sitk.sitkFloat64:
        return sitk.Cast(imagen, sitk.sitkFloat32)
    return sitk.Cast(sitk.Clamp(imagen, lowerBound=-32768, upperBound=32767), sitk.sitkInt16)


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
    lector.SetOutputPixelType(sitk.sitkInt16)        # DICOM de CT: HU enteros; evita una copia de 32 bits
    try:
        return lector.Execute(), len(archivos)
    except RuntimeError as error:
        raise ValueError(f"No pude armar la serie DICOM: {str(error).splitlines()[0][:200]}") from error
