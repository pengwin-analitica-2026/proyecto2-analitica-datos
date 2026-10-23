"""Reducción de artefactos de metal (MAR) por interpolación lineal en el sinograma (LI-MAR). Experimental.

El metal (clavos, prótesis, un fijador externo) absorbe casi todos los rayos X y deja rayas brillantes y
oscuras que atraviesan el corte. En esos cortes el modelo marca tejido como fragmentos de hueso
(ver calidad_entrada.py). El método clásico para atenuarlas (Kalender et al., Radiology 1987):

1. Se proyecta el corte a un sinograma (transformada de Radon): una fila por ángulo de los rayos.
2. Se marca la «traza» del metal: los rayos que pasan por el metal.
3. En cada ángulo se reemplazan esos rayos por una interpolación lineal de los vecinos sin metal.
4. Se reconstruye el corte (retroproyección filtrada) y se vuelve a poner el metal tal cual.

Aquí no tenemos las proyecciones originales del tomógrafo, así que se trabaja sobre las del corte ya
reconstruido: atenúa bien las rayas, pero suaviza un poco el tejido cerca del metal. Con los volúmenes
de prueba del profesor: en G3_P1_A (metal sintético) los fragmentos falsos a más de 20 mm pasaron de 5 a
0; en G3_P1_B (fijador externo real) de 12 a 8. Solo cambia los cortes con metal y nunca los HU del metal.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi

from calidad_entrada import METAL_HU, cortes_con_metal

N_ANGULOS = 180
HILOS = 6        # hilos y no procesos: en AWS Lambda no hay /dev/shm para multiprocessing


def li_mar(corte_hu: np.ndarray, umbral_hu: float = METAL_HU, n_angulos: int = N_ANGULOS) -> np.ndarray:
    """Un corte axial en HU con las rayas del metal atenuadas. Sin metal, lo devuelve igual."""
    from skimage.transform import iradon, radon
    original = corte_hu.astype(np.float32)
    corte = np.clip(original, -1024, None)
    es_metal = corte >= umbral_hu
    if not es_metal.any():
        return original
    traza_metal = ndi.binary_dilation(es_metal, iterations=2)       # incluye el borde borroso del metal
    mu = (corte + 1024) / 1000.0                                     # atenuación relativa: aire = 0
    mu[traza_metal] = 0
    angulos = np.linspace(0, 180, n_angulos, endpoint=False)
    seno = radon(mu, angulos, circle=False)
    traza = radon(traza_metal.astype(np.float32), angulos, circle=False) > 1e-3
    detector = np.arange(seno.shape[0])
    for j in range(seno.shape[1]):
        t = traza[:, j]
        if t.any() and (~t).sum() > 2:
            seno[t, j] = np.interp(detector[t], detector[~t], seno[~t, j])
    corregido = iradon(seno, angulos, circle=False, filter_name="ramp",
                       output_size=corte.shape[0]).astype(np.float32) * 1000.0 - 1024
    fuera = original <= -1024                       # aire y fuera del campo de visión: como venían
    corregido[fuera] = original[fuera]
    corregido[es_metal] = original[es_metal]        # el metal se repone tal cual
    return corregido


def reducir_artefactos_metal(image_hu: np.ndarray, umbral_hu: float = METAL_HU, hilos: int = HILOS,
                             log=print) -> tuple[np.ndarray, dict]:
    """Aplica ``li_mar`` a los cortes con metal. Devuelve (volumen nuevo en float32, resumen)."""
    con_metal, _ = cortes_con_metal(image_hu, umbral_hu)
    cortes = [int(z) for z in np.flatnonzero(con_metal)]
    salida = image_hu.astype(np.float32, copy=True)
    inicio = time.perf_counter()
    if cortes:
        log(f"Reducción de artefactos de metal en {len(cortes)} cortes…")
        with ThreadPoolExecutor(max(1, hilos)) as pool:
            for z, corregido in zip(cortes, pool.map(lambda k: li_mar(image_hu[k], umbral_hu), cortes)):
                salida[z] = corregido
    resumen = {"metodo": "LI-MAR (interpolación lineal en el sinograma)", "umbral_hu": umbral_hu,
               "cortes": cortes, "segundos": round(time.perf_counter() - inicio, 1)}
    return salida, resumen
