"""Aumento geométrico en milímetros reales (corrección del giro de entrenamiento).

El caché estira cada corte a S×S píxeles aunque el original no sea cuadrado ni tenga píxeles
cuadrados, así que un píxel del caché puede medir, por ejemplo, 0.9 mm de ancho y 1.6 mm de alto.
``train_s10.random_affine`` giraba directamente en la grilla de píxeles: con píxeles no cuadrados,
eso no es un giro físico sino un giro más una deformación (la pelvis se "tuerce"), y la red
aprendía con anatomías que no existen.

Aquí la transformación se define en milímetros y se lleva a la grilla de píxeles:

    theta_píxeles = D⁻¹ · A_mm · D,      D = diag(ancho · mm_x, alto · mm_y)

donde ``A_mm`` es la misma combinación de giro, escala y proporción de antes, pero aplicada al
corte en milímetros. Con píxeles cuadrados ``D`` es proporcional a la identidad y el resultado es
idéntico al de ``random_affine`` (prueba en ``tests/test_aumento_fisico.py``), así que los
modelos anteriores siguen siendo reproducibles.

La imagen se remuestrea bilineal y la etiqueta con vecino más cercano (los IDs de fragmento no se
mezclan); las esquinas que quedan vacías son fondo; las cajas de detección se recalculan después a
partir de la etiqueta transformada.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def mm_por_pixel(caso: dict) -> tuple[float, float]:
    """Milímetros por píxel (x, y) de un caso del caché, a partir de su cabecera guardada.

    ``caso`` es lo que devuelve ``build_case_cache.load_case_cache``: trae ``spacing_xyz``
    (mm del volumen original), ``orig_shape`` (z, y, x), ``size`` (lado S) y ``mode``.
    En modo ``stretch`` cada eje se estira por separado; en ``pad`` se rellena a un cuadrado de
    lado max(y, x) en píxeles y luego se redimensiona.
    """
    sx, sy = float(caso["spacing_xyz"][0]), float(caso["spacing_xyz"][1])
    _, alto, ancho = (int(v) for v in caso["orig_shape"])
    lado = int(caso["size"])
    modo = str(caso.get("mode", "stretch"))
    if modo == "pad":
        cuadrado = max(alto, ancho)
        return sx * cuadrado / lado, sy * cuadrado / lado
    return sx * ancho / lado, sy * alto / lado


def theta_mm(angulo: torch.Tensor, escala: torch.Tensor, proporcion: torch.Tensor,
             desplazamiento: torch.Tensor, mm_xy: torch.Tensor | None, alto: int, ancho: int
             ) -> torch.Tensor:
    """Matrices [n, 2, 3] para ``affine_grid`` con el giro definido en milímetros.

    ``mm_xy`` [n, 2] son los mm por píxel (x, y) de cada muestra; None = píxeles cuadrados
    (comportamiento anterior, en la grilla de píxeles).
    """
    n = angulo.shape[0]
    sx, sy = escala * proporcion, escala / proporcion
    cos, sin = torch.cos(angulo), torch.sin(angulo)
    a = torch.zeros(n, 2, 2)
    a[:, 0, 0], a[:, 0, 1] = cos / sx, -sin / sy
    a[:, 1, 0], a[:, 1, 1] = sin / sx, cos / sy
    if mm_xy is not None:
        # Coordenadas normalizadas [-1, 1] → milímetros: multiplicar por (lado/2 · mm). El factor
        # 1/2 se cancela en D⁻¹ A D, así que basta con lado · mm.
        d = torch.stack([ancho * mm_xy[:, 0], alto * mm_xy[:, 1]], dim=1).float()
        a = a * d[:, None, :] / d[:, :, None]          # (D⁻¹ A D)_ij = A_ij · d_j / d_i
    theta = torch.zeros(n, 2, 3)
    theta[:, :, :2] = a
    theta[:, :, 2] = desplazamiento
    return theta


def aplicar_afin(x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Remuestrea imagen (bilineal) y etiqueta (vecino más cercano) con la misma ``theta``."""
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x_out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y_out = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                          align_corners=False)[:, 0].long()
    return x_out, y_out


def random_affine_mm(x: torch.Tensor, y: torch.Tensor, mm_xy: torch.Tensor | None = None,
                     strength: float = 1.0, probability: float = 0.8, wide_rotation: float = 0.0,
                     rotation_degrees: float = 12.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Mismo sorteo que ``train_s10.random_affine``, pero con el giro en milímetros reales.

    ``strength`` = 1: escala 0.80-1.25, proporción 0.85-1.18, rotación ±``rotation_degrees``,
    traslación ±8 % del lado. Con ``mm_xy=None`` es idéntico a la versión anterior.
    """
    n = x.shape[0]
    escala = torch.exp((torch.rand(n) * 2 - 1) * 0.22 * strength)
    proporcion = torch.exp((torch.rand(n) * 2 - 1) * 0.16 * strength)
    angulo = (torch.rand(n) * 2 - 1) * math.radians(rotation_degrees) * strength
    desplazamiento = (torch.rand(n, 2) * 2 - 1) * 0.16 * strength
    if wide_rotation > 0:
        amplio = torch.rand(n) < wide_rotation
        angulo[amplio] = (torch.rand(int(amplio.sum())) * 2 - 1) * math.pi
    sin_tocar = torch.rand(n) >= probability
    escala[sin_tocar], proporcion[sin_tocar] = 1.0, 1.0
    angulo[sin_tocar], desplazamiento[sin_tocar] = 0.0, 0.0
    theta = theta_mm(angulo, escala, proporcion, desplazamiento, mm_xy, x.shape[-2], x.shape[-1])
    return aplicar_afin(x, y, theta)


def mm_de_cortes(datos: dict, indices: np.ndarray) -> torch.Tensor:
    """mm por píxel [n, 2] de los cortes elegidos, a partir de ``load_slices`` (claves mm_x, mm_y)."""
    return torch.from_numpy(np.stack([datos["mm_x"][indices], datos["mm_y"][indices]], axis=1))
