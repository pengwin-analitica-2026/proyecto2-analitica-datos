"""Dashboard PENGWIN (Semana 11): tres visualizadores, métricas y latencia.

Lee las carpetas que deja ``src/demo_export.py`` en ``outputs/demo/<modelo>/<caso>/``. Para correrlo,
desde la raíz del repo:

    pip install -r requirements.txt
    python src/demo_export.py --case 024 --checkpoint outputs/s10/d5_ema_w32_e60_s42.pth --tta
    streamlit run app/dashboard.py

Con ``DEMO_ROOT`` se cambia la carpeta de casos exportados. El botón de inferencia en vivo del
visualizador 2 necesita ``torch`` y el checkpoint; el resto solo lee archivos.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from demo_render import (BASE_RGB, NON_CLINICAL, REGION_LABELS, REGION_ORDER, REGION_STARTS, bone_figure,
                         distance_text, hex_color, legend_items, local_name, mesh_figure, render_slice)

DEMO_ROOT = Path(os.environ.get("DEMO_ROOT", ROOT / "outputs" / "demo"))
CHECKPOINT_DIR = ROOT / "outputs" / "s10"

st.set_page_config(page_title="PENGWIN · fracturas pélvicas", page_icon="🦴", layout="wide")


def discover_cases() -> dict[str, Path]:
    cases = {}
    for meta in sorted(DEMO_ROOT.glob("*/*/meta.json")):
        cases[f"{meta.parent.parent.name} · caso {meta.parent.name}"] = meta.parent
    return cases


@st.cache_resource(show_spinner="Cargando caso…", max_entries=2)
def load_case(path: str) -> dict:
    folder = Path(path)
    data = {"meta": json.loads((folder / "meta.json").read_text())}
    with np.load(folder / "volume.npz") as volume:
        data.update({key: volume[key] for key in volume.files})
    with np.load(folder / "meshes.npz") as meshes:
        data["meshes"] = {key: meshes[key] for key in meshes.files}
    raw = folder / "raw_view.npz"
    if raw.exists():
        with np.load(raw) as view:
            data["raw"] = {key: view[key] for key in view.files}
    return data


def setting(name: str) -> str | None:
    """Variable de entorno o secreto de Streamlit (Streamlit Cloud); None si no existe."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets[name]) if name in st.secrets else None
    except Exception:                       # no hay archivo de secretos
        return None


def remote_predict_slice(url: str, key: str | None, image_u8: np.ndarray, timeout: float = 120.0) -> dict:
    """Manda un corte al servicio de inferencia (api/server.py) y devuelve lo mismo que predict_slice."""
    import base64
    import io
    import urllib.error
    import urllib.request
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(image_u8, dtype=np.uint8), allow_pickle=False)
    request = urllib.request.Request(url.rstrip("/") + "/predict_slice", data=buffer.getvalue(), method="POST",
                                     headers={"Content-Type": "application/octet-stream", "X-API-Key": key or ""})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"el servicio respondió {error.code}: {error.read()[:200].decode(errors='replace')}") from error
    round_trip = (time.perf_counter() - started) * 1000.0
    with np.load(io.BytesIO(base64.b64decode(payload["semantic_npz_b64"]))) as packed:
        semantic = packed["semantic"]
    return {"semantic": semantic[None], "boxes_refined": [payload["boxes_refined"]],
            "boxes_raw": [payload["boxes_raw"]], "latency_ms": payload["latency_ms"],
            "device": payload["device"], "round_trip_ms": round_trip}


@st.cache_resource(show_spinner="Cargando el modelo…")
def load_live_model(checkpoint: str, tta: bool):
    from predict_volume import load_pipeline
    return load_pipeline(checkpoint, tta=tta)


def legend_html(fragments: list[dict]) -> str:
    chips = "".join(f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
                    f'<span style="width:14px;height:14px;background:{color};border-radius:3px;'
                    f'margin-right:6px;display:inline-block;"></span>{label}</span>'
                    for color, label in legend_items(fragments))
    return f'<div style="margin:4px 0 10px 0;">{chips}<span style="opacity:.7">· los secundarios son tintes más claros con contorno blanco</span></div>'


def fragments_dataframe(rows: list[dict]):
    import pandas as pd
    return pd.DataFrame([{"Hueso": REGION_LABELS[r["region"]], "Fragmento": local_name(r["id"]),
                          "Distancia al principal": "—" if r["is_main"] else distance_text(r["distance_mm"]),
                          "Volumen (mm³)": round(r["volume_mm3"]), "Cortes": f"{r['z_range'][0]}–{r['z_range'][1]}"}
                         for r in rows])


# ----------------------------------------------------------------------------- encabezado
st.title("Fracturas pélvicas en CT")
st.caption("Detección, segmentación por fragmentos y distancia de separación en milímetros · dataset PENGWIN")
st.error(NON_CLINICAL, icon="⚠️")

cases = discover_cases()
if not cases:
    st.info(f"No hay casos exportados en `{DEMO_ROOT}`. Genera uno con:\n\n"
            "`python src/demo_export.py --case 024 --checkpoint outputs/s10/d5_ema_w32_e60_s42.pth --tta`")
    st.stop()

with st.sidebar:
    st.header("Caso")
    choice = st.selectbox("Modelo y caso exportado", list(cases))
    data = load_case(str(cases[choice]))
    meta = data["meta"]
    st.caption(f"{meta['n_slices']} cortes · {meta['size']}×{meta['size']} px · spacing del header "
               f"{meta['spacing_native_xyz'][0]:.2f}×{meta['spacing_native_xyz'][1]:.2f}×"
               f"{meta['spacing_native_xyz'][2]:.2f} mm")
    st.caption(f"Modelo {meta['model']['checkpoint']} · {meta['model']['parameters'] / 1e6:.2f} M parámetros "
               f"· TTA {'sí' if meta['tta'] else 'no'}")
    st.markdown(legend_html(meta["fragments_pred"]), unsafe_allow_html=True)

provisional = str(meta.get("distance_method", "")).startswith("provisional")
all_fragments = meta["fragments_pred"]
spacing = tuple(meta["spacing_cache_xyz"])
has_gt = data.get("gt_ids") is not None

with st.sidebar:
    min_mm3 = st.slider("Volumen mínimo del fragmento (mm³)", 0, 3000, 1000, 100,
                        help="Oculta los fragmentos secundarios más chicos que esto: casi siempre son ruido de la "
                             "segmentación. El fragmento principal nunca se oculta.")
fragments = [r for r in all_fragments if r["is_main"] or r["volume_mm3"] >= min_mm3]
hidden = len(all_fragments) - len(fragments)
visible_ids = np.zeros(256, dtype=bool)
visible_ids[[r["id"] for r in fragments]] = True
with st.sidebar:
    st.caption(f"{len(fragments)} fragmentos visibles" + (f" · {hidden} ocultos por tamaño" if hidden else ""))
    if has_gt:
        real = meta["fragments_gt"]
        st.caption(f"Ground truth: {len(real)} fragmentos ({sum(1 for r in real if not r['is_main'])} secundarios)")

tab1, tab2, tab3, tab4 = st.tabs(["1 · Volumen crudo", "2 · Corte por corte", "3 · Reconstrucción 3D",
                                  "Métricas y latencia"])

# ----------------------------------------------------------------------------- visualizador 1
with tab1:
    st.subheader("Volumen crudo, solo hueso")
    st.caption("Sin modelo: umbral de 250 HU sobre el volumen sin procesar, limpieza de camilla y cables, "
               "y proyección de máxima intensidad (MIP) en los tres ejes.")
    raw = data.get("raw")
    if raw is None:
        st.info("Este caso se exportó sin la vista cruda (`--no-raw-view`).")
    else:
        cols = st.columns(3)
        for col, (key, name) in zip(cols, (("mip_axial", "Axial"), ("mip_coronal", "Coronal"),
                                           ("mip_sagital", "Sagital"))):
            col.image(raw[key], caption=f"MIP {name}", use_container_width=True)
        if st.toggle("Mostrar isosuperficie 3D del hueso", value=False,
                     help="Malla rotable de la misma máscara ósea; pesa más que la MIP."):
            st.plotly_chart(bone_figure(raw["bone_verts_zyx"], raw["bone_faces"]), use_container_width=True)

# ----------------------------------------------------------------------------- visualizador 2
with tab2:
    st.subheader("Inferencia corte por corte")
    first, last = meta["bone_z_range"] or [0, meta["n_slices"] - 1]
    left, right = st.columns([3, 2])
    with right:
        z = st.slider("Corte", 0, meta["n_slices"] - 1, (first + last) // 2,
                      help=f"El hueso aparece entre los cortes {first} y {last}.")
        show_masks = st.checkbox("Máscaras por macro-hueso", True)
        show_boxes = st.checkbox("Cajas de detección", True)
        box_kind = st.radio("Caja", ["refined", "raw"], horizontal=True,
                            format_func=lambda k: "refinada con la semántica" if k == "refined" else "cabeza grid cruda")
        show_gt = st.checkbox("Contorno del ground truth", False, disabled=not has_gt)
        slice_meta = meta["slices"][z]
        st.markdown("**Presencia por región en este corte**")
        for name, prob in zip(REGION_ORDER, slice_meta["classification"]):
            st.progress(float(prob), text=f"{REGION_LABELS[name]} · {prob:.2f}")
    with left:
        ids_slice = np.where(visible_ids[data["ids"][z]], data["ids"][z], 0)
        image = render_slice(data["images"][z], ids_slice, fragments, slice_meta,
                             gt_slice=data["gt_ids"][z] if has_gt else None, boxes=box_kind,
                             show_masks=show_masks, show_boxes=show_boxes, show_gt=show_gt,
                             title=f"corte {z + 1}/{meta['n_slices']}")
        st.image(image, use_container_width=True)
    st.caption("La distancia junto a cada fragmento es la mínima borde a borde, en mm, entre ese fragmento y el "
               "principal de su hueso, medida sobre el volumen 3D completo (spacing del header): es la misma en "
               "todos los cortes donde el fragmento aparece.")
    if provisional:
        st.warning("Distancia con método provisional: ver la pestaña de métricas.")
    st.dataframe(fragments_dataframe(fragments), use_container_width=True, hide_index=True)

    with st.expander("Inferir este corte en vivo (modelo real, con latencia)"):
        remote_url = setting("INFERENCE_URL")
        checkpoint = CHECKPOINT_DIR / meta["model"]["checkpoint"]
        if not remote_url and not checkpoint.exists():
            st.info("La inferencia en vivo no está configurada en esta instalación: falta el servicio de "
                    "inferencia (`INFERENCE_URL`) o el checkpoint en `outputs/s10/`.")
        elif st.button("Inferir el corte seleccionado"):
            try:
                if remote_url:
                    live = remote_predict_slice(remote_url, setting("INFERENCE_KEY"), data["images"][z])
                    where = f"{live['device']} del servicio remoto · ida y vuelta {live['round_trip_ms']:.0f} ms"
                else:
                    from predict_volume import predict_slice
                    model, device, info = load_live_model(str(checkpoint), False)
                    predict_slice(model, device, data["images"][z])     # pasada de calentamiento
                    live = predict_slice(model, device, data["images"][z])
                    where = f"{device} local"
                semantic = live["semantic"][0]
                starts = np.array([0] + [REGION_STARTS[r] for r in REGION_ORDER], dtype=np.uint8)
                live_ids = starts[semantic]
                live_image = render_slice(data["images"][z], live_ids, [], live["boxes_refined"][0] and
                                          {"boxes_refined": live["boxes_refined"][0],
                                           "boxes_raw": live["boxes_raw"][0]}, title=f"EN VIVO · corte {z + 1}")
                a, b = st.columns([3, 2])
                a.image(live_image, use_container_width=True)
                b.metric("Latencia (modelo + NMS)", f"{live['latency_ms']:.0f} ms")
                b.caption(f"Dispositivo: {where} · sin TTA · un corte {meta['size']}×{meta['size']}. "
                          "En vivo se muestra la región anatómica y las cajas; la separación de fragmentos y la "
                          "distancia en mm usan el volumen completo y vienen de la exportación.")
            except Exception as error:          # sin torch ni servicio, o el servicio está dormido
                st.error(f"No se pudo inferir en vivo: {error}")

# ----------------------------------------------------------------------------- visualizador 3
with tab3:
    st.subheader("Reconstrucción 3D final")
    st.caption("Máscaras predichas apiladas, marching cubes por fragmento y malla interactiva. Cada fragmento "
               "conserva el color de su macro-hueso y lleva su ID y su distancia al principal. Arrastra para rotar.")
    c1, c2, c3 = st.columns(3)
    show_main = c1.checkbox("Mostrar el fragmento principal", True)
    main_opacity = c2.slider("Opacidad del principal", 0.1, 1.0, 0.4, 0.05)
    show_gt3 = c3.checkbox("Superponer ground truth", False, disabled=not has_gt)
    label_contacts = st.checkbox("Rotular también los fragmentos en contacto (0.0 mm)", True)
    fig = mesh_figure(data["meshes"], fragments, spacing, show_main=show_main, main_opacity=main_opacity,
                      show_gt=show_gt3, gt_fragments=meta["fragments_gt"], label_contacts=label_contacts)
    st.plotly_chart(fig, use_container_width=True)
    if provisional:
        st.warning("Distancia con método provisional: ver la pestaña de métricas.")

# ----------------------------------------------------------------------------- métricas
with tab4:
    st.subheader("Distancia de separación: predicción contra ground truth")
    st.caption(f"Método de la distancia: {meta.get('distance_method', 'desconocido')}")
    if provisional:
        st.warning("El volumen de IDs se arma conservando solo los vóxeles que la cabeza de instancia etiquetó, "
                   "porque la corrección `fill_unlabeled` de `postprocess_3d` todavía no está en este repo. "
                   "Volver a exportar los casos cuando llegue.")
    comparison = meta.get("comparison")
    if comparison:
        import pandas as pd
        def mm(value):
            return "—" if value is None else f"{value:.1f}"

        table = pd.DataFrame([{"Hueso": REGION_LABELS[r["region"]], "Fragmento real": local_name(r["gt_id"]),
                               "Distancia real (mm)": mm(r["gt_mm"]),
                               "Fragmento predicho": "no detectado" if r["pred_id"] is None else local_name(r["pred_id"]),
                               "Distancia predicha (mm)": mm(r["pred_mm"]),
                               "Error absoluto (mm)": mm(r["abs_error_mm"]),
                               "Dice 3D": f"{r['dice3d']:.2f}"} for r in comparison])
        st.dataframe(table, use_container_width=True, hide_index=True)
        paired = [r for r in comparison if r["abs_error_mm"] is not None]
        apart = [r for r in paired if r["gt_mm"] >= 0.05]
        m1, m2, m3 = st.columns(3)
        m1.metric("Fragmentos secundarios reales", len(comparison),
                  help="Los que no tienen pareja predicha aparecen como «no detectado».")
        m2.metric("Error medio, todos los emparejados", f"{np.mean([r['abs_error_mm'] for r in paired]):.1f} mm"
                  if paired else "—", help=f"{len(paired)} fragmentos; incluye los que están en contacto (0.0 mm).")
        m3.metric("Error medio, solo separados (> 0 mm)", f"{np.mean([r['abs_error_mm'] for r in apart]):.1f} mm"
                  if apart else "—", help=f"{len(apart)} fragmentos: es la medida que importa para la separación.")
    else:
        st.info("Este caso no tiene ground truth.")

    st.subheader("Latencia por imagen")
    st.metric("Inferencia del volumen al exportar", f"{meta['ms_per_slice']:.0f} ms por corte",
              help=f"{meta['device']} · TTA {'sí' if meta['tta'] else 'no'} · {meta['n_slices']} cortes")
    latency = DEMO_ROOT / "latency.json"
    if latency.exists():
        import pandas as pd
        st.dataframe(pd.DataFrame(json.loads(latency.read_text())), use_container_width=True, hide_index=True)
    else:
        st.caption("Aún no hay mediciones de latencia en CPU y GPU (`src/measure_latency.py`).")
