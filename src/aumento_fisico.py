"""Aumentos geométricos compartidos por Semana 9 y Semana 10.

La matriz se construye en coordenadas físicas cuando se proporciona ``spacing_yx``. De esta
manera una rotación no deforma la anatomía al trabajar con píxeles no cuadrados. Imagen y todos
los canales 2.5D usan interpolación bilineal; los IDs de la máscara usan vecino más cercano.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def random_affine(x: torch.Tensor, y: torch.Tensor, strength: float = 1.0,
                  probability: float = 0.8, wide_rotation: float = 0.0,
                  rotation_degrees: float = 12.0,
                  spacing_yx: torch.Tensor | np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Escala, proporción, rotación y traslación coherentes entre entrada y máscara.

    ``spacing_yx`` contiene milímetros por píxel para cada muestra. Cuando se pasa, los términos
    cruzados de la rotación compensan la relación física entre ambos ejes. Si se omite se conserva
    exactamente el comportamiento histórico en coordenadas de imagen.
    """
    n = x.shape[0]
    scale = torch.exp((torch.rand(n) * 2 - 1) * 0.22 * strength)
    aspect = torch.exp((torch.rand(n) * 2 - 1) * 0.16 * strength)
    angle = (torch.rand(n) * 2 - 1) * math.radians(rotation_degrees) * strength
    shift = (torch.rand(n, 2) * 2 - 1) * 0.16 * strength
    if wide_rotation > 0:
        wide = torch.rand(n) < wide_rotation
        angle[wide] = (torch.rand(int(wide.sum())) * 2 - 1) * math.pi
    keep = torch.rand(n) >= probability
    scale[keep], aspect[keep], angle[keep], shift[keep] = 1.0, 1.0, 0.0, 0.0
    sx, sy = scale * aspect, scale / aspect
    cos, sin = torch.cos(angle), torch.sin(angle)
    if spacing_yx is None:
        physical_y_over_x = torch.ones(n)
    else:
        spacing = torch.as_tensor(spacing_yx, dtype=torch.float32, device="cpu")
        if spacing.shape != (n, 2):
            raise ValueError(f"spacing_yx debe tener forma ({n}, 2), no {tuple(spacing.shape)}")
        physical_y_over_x = (spacing[:, 0] * x.shape[-2]) / (spacing[:, 1] * x.shape[-1])

    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos / sx
    theta[:, 0, 1] = -sin * physical_y_over_x / sy
    theta[:, 0, 2] = shift[:, 0]
    theta[:, 1, 0] = sin / physical_y_over_x / sx
    theta[:, 1, 1] = cos / sy
    theta[:, 1, 2] = shift[:, 1]
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x_out = torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y_out = torch.nn.functional.grid_sample(
        y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    return x_out, y_out
