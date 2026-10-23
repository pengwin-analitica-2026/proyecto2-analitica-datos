"""Controles de calidad de un CT antes de confiar en sus medidas: espaciado de la cabecera y metal.

Dos problemas que el modelo no puede ver solo, encontrados con los volúmenes de prueba del profesor:

1. **Espaciado imposible en la cabecera.** Un .mha decía ``ElementSpacing = 0.938 0.938 100``: 100 mm
   entre cortes, aunque los cortes vecinos son casi idénticos (en un CT real van de 0.5 a 5 mm). La red
   trabaja corte a corte y segmenta bien, pero todo lo que se mide en mm sale mal: las distancias entre
   cortes se multiplican, el vóxel pesa 100 veces más (los umbrales en mm³ dejan de filtrar ruido) y las
   vistas 3D salen aplastadas. ``revisar_espaciado`` lo detecta y la subida deja corregir el grosor.
2. **Metal.** Un fijador externo produce rayas de artefacto en los cortes donde está; ahí el modelo
   puede marcar tejido como fragmentos de hueso, con distancias de varios centímetros que no son
   reales. PENGWIN no trae casos así (el que más metal tiene, 027, llega a 21 810 mm³).
   ``revisar_metal`` encuentra esos cortes y ``marcar_fragmentos_con_metal`` señala los fragmentos que
   caen en ellos, sin borrarlos: la decisión clínica no es nuestra.

Todo usa solo la imagen y su cabecera; vale igual en la página de AWS, en Streamlit y al exportar.
"""
from __future__ import annotations

import numpy as np

# Rango plausible de espaciado en un CT, en mm. Entre cortes: 0.3 (alta resolución) a 10 (cortes muy
# gruesos de un estudio antiguo); en el plano: 0.2 a 2.5. Fuera de eso la cabecera casi seguro está mal.
ENTRE_CORTES_MM = (0.3, 10.0)
EN_PLANO_MM = (0.2, 2.5)
METAL_HU = 2500            # el hueso cortical no pasa de ~2000 HU; desde aquí es metal (incluye 3000 exactos)
METAL_MIN_VOXELES = 20     # vóxeles de metal para contar un corte como «con metal»
METAL_AVISO_MM3 = 1000.0   # metal total a partir del cual se avisa (implantes pequeños no afectan)
FRACCION_CORTES = 0.5      # un fragmento se marca si la mitad o más de sus cortes tienen metal


def revisar_espaciado(spacing_xyz) -> list[dict]:
    """Avisos si el espaciado de la cabecera no es el de un CT. Lista vacía si es plausible."""
    sx, sy, sz = (float(v) for v in spacing_xyz)
    avisos = []
    if not ENTRE_CORTES_MM[0] <= sz <= ENTRE_CORTES_MM[1]:
        avisos.append({
            "tipo": "espaciado",
            "titulo": f"Distancia entre cortes sospechosa: {sz:g} mm",
            "texto": (f"La cabecera dice {sz:g} mm entre cortes; en un CT suele estar entre 0.5 y 5 mm. "
                      "La segmentación de cada corte no cambia, pero las distancias en mm, los volúmenes "
                      "y las vistas 3D sí dependen de este valor y no son confiables. Si conoces el grosor "
                      "real (está en los datos del estudio), vuelve a subir el CT indicándolo en "
                      "«Grosor entre cortes».")})
    if not (EN_PLANO_MM[0] <= sx <= EN_PLANO_MM[1] and EN_PLANO_MM[0] <= sy <= EN_PLANO_MM[1]):
        avisos.append({
            "tipo": "espaciado",
            "titulo": f"Tamaño de píxel sospechoso: {sx:g} × {sy:g} mm",
            "texto": ("En un CT el píxel mide entre 0.2 y 2.5 mm. Con este valor las medidas en mm no son "
                      "confiables; revisa la cabecera del archivo.")})
    return avisos


def cortes_con_metal(image_hu: np.ndarray, umbral_hu: float = METAL_HU,
                     min_voxeles: int = METAL_MIN_VOXELES) -> tuple[np.ndarray, np.ndarray]:
    """(corte con metal [z] bool, vóxeles de metal por corte [z]). Corte a corte, sin copiar el volumen."""
    conteo = np.array([int(np.count_nonzero(corte >= umbral_hu)) for corte in image_hu], dtype=np.int64)
    return conteo >= min_voxeles, conteo


def revisar_metal(image_hu: np.ndarray, spacing_xyz, umbral_hu: float = METAL_HU,
                  aviso_mm3: float = METAL_AVISO_MM3) -> dict:
    """Resumen del metal del volumen: cortes afectados, volumen en mm³ y si amerita aviso."""
    con_metal, conteo = cortes_con_metal(image_hu, umbral_hu)
    volumen = float(conteo.sum() * np.prod([float(v) for v in spacing_xyz]))
    z = np.flatnonzero(con_metal)
    return {"umbral_hu": umbral_hu, "volumen_mm3": round(volumen, 1), "n_cortes": int(len(z)),
            "cortes": [int(v) for v in z], "rango_z": [int(z[0]), int(z[-1])] if len(z) else None,
            "avisar": bool(volumen >= aviso_mm3 and len(z))}


def marcar_fragmentos_con_metal(rows: list[dict], cortes_metal, fraccion: float = FRACCION_CORTES) -> int:
    """Agrega ``metal`` (bool) a cada fila de ``fragment_table``; devuelve cuántos secundarios se marcaron.

    Un secundario se marca si al menos ``fraccion`` de los cortes que ocupa tienen metal. El principal
    nunca se marca: es el hueso grande y su presencia no depende de un artefacto.
    """
    metal = set(int(z) for z in cortes_metal)
    marcados = 0
    for row in rows:
        z0, z1 = row["z_range"]
        cortes = range(int(z0), int(z1) + 1)
        afectado = (not row.get("is_main")) and bool(metal) and \
            sum(z in metal for z in cortes) >= fraccion * len(cortes)
        row["metal"] = bool(afectado)
        marcados += afectado
    return marcados


def aviso_metal(resumen: dict, marcados: int) -> dict | None:
    """El aviso para mostrar, o None si el metal es poco o no hay."""
    if not resumen.get("avisar"):
        return None
    z0, z1 = resumen["rango_z"]
    volumen = f"{resumen['volumen_mm3']:,.0f}".replace(",", ".")          # separador de miles en español
    texto = (f"Hay unos {volumen} mm³ de metal ({resumen['umbral_hu']} HU o más) en "
             f"{resumen['n_cortes']} cortes, entre el {z0} y el {z1}. Las rayas de artefacto del metal pueden "
             "hacer que el modelo marque tejido como fragmentos de hueso. ")
    texto += (f"{marcados} fragmento{'s' if marcados != 1 else ''} en esos cortes "
              f"{'quedan marcados' if marcados != 1 else 'queda marcado'} con ⚠: su existencia y su distancia "
              "no son confiables." if marcados else "Ningún fragmento secundario cae en esos cortes.")
    return {"tipo": "metal", "titulo": "CT con metal", "texto": texto}
