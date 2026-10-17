"""Vistas del preprocesamiento paso a paso, para la pestaña «Preprocesamiento» del dashboard.

Antes de entrar al modelo, un volumen pasa por cuatro etapas. Cada una se guarda como dos imágenes
pequeñas (el corte axial con más hueso y la proyección de máxima intensidad axial) y un texto que dice
qué se hizo:

1. ``crudo``: el arreglo tal como viene en el archivo (``GetArrayFromImage``, sin aplicar la
   dirección de la cabecera), con la ``TransformMatrix`` que trae.
2. ``orientado``: después de la orientación con los metadatos. Si la cabecera dice que el volumen
   está guardado girado 180° (LPI), aquí se ve enderezado; si no, queda igual.
3. ``giro_axial``: después de enderezar el giro del paciente dentro del escáner (simetría de la
   pelvis), si fue claro.
4. ``entrada``: lo que recibe la red: ventana ósea (−500 a 1300 HU), gamma 1.5 y 384×384 píxeles.

Las etapas 1 a 3 se muestran con la misma ventana ósea sin gamma y con la proporción física del
vóxel, para que solo cambie lo que hizo la etapa. La 4 se muestra tal cual entra al modelo (estirada a
384×384). Se guardan en ``pasos.npz`` (``<clave>_corte`` y ``<clave>_mip``) y el texto en
``meta.json`` (``pasos``).
"""
from __future__ import annotations

import numpy as np
from skimage.transform import resize

from build_case_cache import to_uint8
from pengwin_io import BONE_HU_THRESHOLD, window_hu

LADO = 384


def corte_con_mas_hueso(image_hu: np.ndarray) -> int:
    """Índice z del corte con más vóxeles sobre el umbral óseo (suele mostrar las alas ilíacas)."""
    paso = max(1, image_hu.shape[1] // 128)
    conteo = (image_hu[:, ::paso, ::paso] > BONE_HU_THRESHOLD).sum(axis=(1, 2))
    return int(np.argmax(conteo)) if conteo.max() > 0 else len(image_hu) // 2


def _a_pantalla(plano_hu: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """HU → uint8 con ventana ósea sin gamma, redimensionado a la proporción física (lado mayor 384)."""
    imagen = window_hu(plano_hu.astype(np.float32), gamma=1.0)
    alto, ancho = plano_hu.shape[0] * sy, plano_hu.shape[1] * sx
    escala = LADO / max(alto, ancho)
    forma = (max(1, round(alto * escala)), max(1, round(ancho * escala)))
    return to_uint8(resize(imagen, forma, order=1, preserve_range=True, anti_aliasing=True))


def vista_hu(image_hu: np.ndarray, spacing_xyz, z: int | None = None) -> tuple[dict[str, np.ndarray], int]:
    """Corte axial ``z`` (el de más hueso si es None) y MIP axial de un volumen en HU."""
    z = corte_con_mas_hueso(image_hu) if z is None or not 0 <= z < len(image_hu) else z
    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    paso = max(1, len(image_hu) // 200)                  # la MIP no necesita todos los cortes
    mip = image_hu[::paso].max(axis=0)
    return {"corte": _a_pantalla(image_hu[z], sy, sx), "mip": _a_pantalla(mip, sy, sx)}, z


def vista_entrada(images_u8: np.ndarray, z: int) -> dict[str, np.ndarray]:
    """Lo que entra a la red: el corte ``z`` ya preprocesado (384×384) y la MIP de esos cortes."""
    z = min(max(z, 0), len(images_u8) - 1)
    return {"corte": images_u8[z].copy(), "mip": images_u8.max(axis=0)}


class Pasos:
    """Acumula las etapas: ``agregar`` guarda las imágenes y el texto de cada una."""

    def __init__(self) -> None:
        self.textos: list[dict] = []
        self.arreglos: dict[str, np.ndarray] = {}
        self.z: int | None = None

    def agregar_hu(self, clave: str, titulo: str, texto: str, image_hu: np.ndarray, spacing_xyz,
                   **extra) -> None:
        vista, self.z = vista_hu(image_hu, spacing_xyz, self.z)
        self._guardar(clave, titulo, texto, vista, forma=list(image_hu.shape), **extra)

    def agregar_entrada(self, titulo: str, texto: str, images_u8: np.ndarray) -> None:
        self._guardar("entrada", titulo, texto, vista_entrada(images_u8, self.z or len(images_u8) // 2),
                      forma=list(images_u8.shape))

    def _guardar(self, clave: str, titulo: str, texto: str, vista: dict, **extra) -> None:
        self.textos.append({"clave": clave, "titulo": titulo, "texto": texto, "z": self.z, **extra})
        for nombre, arreglo in vista.items():
            self.arreglos[f"{clave}_{nombre}"] = arreglo


def texto_matriz(direccion) -> str:
    valores = [round(float(v)) if abs(float(v) - round(float(v))) < 1e-3 else round(float(v), 3)
               for v in direccion]
    return " ".join(str(v) for v in valores)
