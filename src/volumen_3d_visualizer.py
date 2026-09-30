"""
volumen_3d_visualizer.py
-------------------------
Visualizador 1 (V1) — reconstruccion 3D interactiva del volumen crudo.

Reemplaza la version anterior de mip_visualizer.py (tres PNG estaticos
axial/coronal/sagital) por una isosuperficie 3D real, rotable con el mouse,
que es lo que pide el pliego para el V1: "Reconstruccion 3D inicial del
volumen original... mostrando UNICAMENTE el hueso".

No interviene el modelo: es preprocesamiento clasico (umbral HU + silueta
corporal), igual que exige la seccion 3.4 del proyecto.

Limpieza aplicada (clean_bone_mask de pengwin_io.py):
1. Silueta del cuerpo del paciente: componente conectado mas grande de la
   mascara HU > -300, con huecos internos rellenos. Los rieles de la
   camilla y los cables quedan fuera de esta silueta porque estan
   separados del cuerpo por aire, aunque tengan HU alto igual que el hueso.
   (Un filtro por tamano de componente NO sirve aqui: un riel recto puede
   pesar mas voxeles que un fragmento oseo fracturado.)
2. Interseccion con el umbral oseo (250 HU) y descarte de motas aisladas
   pequenas dentro del cuerpo (artefactos puntuales de metal).
3. Filtro por profundidad: los componentes pequenos que tocan la zona a
   menos de 10 mm de la piel (cinturones, electrodos o cables que rozan la
   silueta) se descartan. Un fragmento oseo real queda rodeado de tejido
   blando. Los componentes grandes (esqueleto, femures) se conservan siempre.
4. Recuperacion de hueso tenue (150-250 HU) y relleno del interior esponjoso,
   partiendo de la mascara ya limpia y solo a >= 10 mm de la piel.

Si aun asi se ve algun cable pegado al cuerpo o falta hueso muy periferico,
ajusta BODY_HU_THRESHOLD, MIN_NOISE_VOXELS o MIN_SURFACE_DEPTH_MM en
pengwin_io.py; ese modulo es compartido con el visualizador 3.

Requisitos:
    pip install SimpleITK numpy scipy scikit-image plotly

Uso (desde la raiz del repo):
    python src/volumen_3d_visualizer.py                 # corre sobre CASOS_EJEMPLO
    python src/volumen_3d_visualizer.py 007              # un caso puntual
"""

import sys
from pathlib import Path

import numpy as np
from skimage import measure
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pengwin_io import load_case, clean_bone_mask, bone_mask, BONE_HU_THRESHOLD  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CASOS_EJEMPLO = ["001", "025", "068"]

# --- rendimiento: sin downsample, un volumen 400x512x512 genera demasiados
# triangulos para un HTML interactivo fluido. factor 2 = medio detalle,
# suficiente para inspeccion visual del V1 (no es la reconstruccion final
# con fragmentos, esa es el V3 y se hace sobre las mascaras predichas).
DOWNSAMPLE_FACTOR = 2


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask
    return mask[::factor, ::factor, ::factor]


def mask_to_mesh(mask: np.ndarray, spacing_zyx: tuple):
    """Marching cubes sobre la mascara binaria. Devuelve vertices y caras en mm."""
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=spacing_zyx
    )
    return verts, faces


def render_html(verts, faces, case_id: str, n_voxels_finales: int, n_voxels_totales: int):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],  # (z,y,x) -> ejes x,y,z
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="ivory", opacity=1.0,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300),
        )
    ])
    fig.update_layout(
        title=(
            f"Caso {case_id} — Volumen crudo (V1), solo hueso "
            f"(umbral {BONE_HU_THRESHOLD} HU)<br>"
            f"<sub>{n_voxels_finales:,} voxeles oseos tras limpieza por silueta y "
            f"profundidad y recuperacion de hueso tenue — sin intervencion del modelo</sub>"
        ),
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    out_path = OUTPUT_DIR / f"volumen_3d_{case_id}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Guardado: {out_path}")


def procesar_caso(case_id: str):
    case_id = case_id.zfill(3)
    try:
        caso = load_case(case_id, load_label=False)
    except FileNotFoundError:
        print(f"[AVISO] No se encontro el caso {case_id}, se omite.")
        return

    # spacing_xyz de SimpleITK es (sx, sy, sz); el volumen esta en (z,y,x).
    sx, sy, sz = caso.spacing_xyz

    mask_cruda = bone_mask(caso.image)
    mask_limpia = clean_bone_mask(caso.image, spacing_zyx=(sz, sy, sx))

    n_totales = int(mask_cruda.sum())
    n_finales = int(mask_limpia.sum())
    print(f"Caso {case_id}: {n_totales:,} voxeles > {BONE_HU_THRESHOLD} HU en todo el volumen -> "
          f"{n_finales:,} en la mascara final")

    mask_ds = downsample_mask(mask_limpia, DOWNSAMPLE_FACTOR)
    if mask_ds.sum() == 0:
        print(f"[AVISO] Caso {case_id}: mascara vacia tras limpieza/downsample, se omite.")
        return

    # el downsample multiplica el espaciado efectivo entre voxeles.
    spacing_zyx = (sz * DOWNSAMPLE_FACTOR, sy * DOWNSAMPLE_FACTOR, sx * DOWNSAMPLE_FACTOR)

    verts, faces = mask_to_mesh(mask_ds, spacing_zyx)
    render_html(verts, faces, case_id, n_finales, n_totales)


def main():
    casos = sys.argv[1:] if len(sys.argv) > 1 else CASOS_EJEMPLO
    for case_id in casos:
        procesar_caso(case_id)


if __name__ == "__main__":
    main()