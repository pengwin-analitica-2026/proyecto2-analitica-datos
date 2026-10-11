"""Dashboard PENGWIN (Semana 11): tres visualizadores, métricas y latencia.

Lee las carpetas que deja ``src/demo_export.py`` en ``outputs/demo/<modelo>/<caso>/``. Para correrlo,
desde la raíz del repo:

    pip install -r requirements.txt
    python src/demo_export.py --case 024 044 --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth --tta
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
RESULTS_DIR = ROOT / "outputs" / "s10"          # JSON de la evaluación en test (Semana 10)

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
            "device": payload["device"], "round_trip_ms": round_trip, "model": payload.get("model")}


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


def read_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


TEST_ROWS = (("F1 de clasificación", "classification_f1_macro", 0.85),
             ("AUC de clasificación", "classification_auc_macro", 0.85),
             ("IoU medio de detección", "detection_iou_mean", 0.65),
             ("mAP50", "detection_map50", 0.65),
             ("mAP50:95", "detection_map50_95", 0.40),
             ("Dice de segmentación semántica", "segmentation_dice_macro", 0.85),
             ("IoU de segmentación semántica", "segmentation_iou_macro", 0.70),
             ("Dice por fragmento", "instance_dice_matched", 0.85),
             ("IoU por fragmento", "instance_iou_matched", 0.70))


def test_metrics_section() -> None:
    """Las dos evaluaciones en test de la Semana 10, leídas de outputs/s10 (no se recalcula nada)."""
    import pandas as pd
    first, second = read_json("TEST_d5_s42_tta.json"), read_json("TEST2_h6_s42_tta.json")
    if not first or not second:
        return
    a, b = first["subsets"]["todo"], second["subsets"]["todo"]
    slices = f"{b['n_slices']:,}".replace(",", " ")
    st.subheader(f"Resultados en test ({b['n_cases']} casos, {slices} cortes)")
    st.dataframe(pd.DataFrame([{"Métrica": name, "Objetivo": f"{goal:.2f}",
                                "1.ª evaluación (D5)": f"{a[key]:.3f}", "2.ª evaluación (H6)": f"{b[key]:.3f}",
                                "Cumple (H6)": "sí" if b[key] >= goal else "no"} for name, key, goal in TEST_ROWS]),
                 use_container_width=True, hide_index=True)
    st.caption("**Cómo leer las dos columnas.** La primera evaluación (D5) se hizo con la configuración congelada "
               "antes de ver test: es la cifra limpia, y no cumplía el Dice por fragmento (0.845). La segunda (H6, "
               "el modelo de esta demo) se hizo después de analizar los fallos de la primera, y ese análisis miró "
               "dos casos de test (012 y 039). H6 se eligió solo con validación y pruebas de estrés; sus tres "
               "semillas dan 0.875, 0.875 y 0.876 en Dice por fragmento.")
    c1, c2 = st.columns(2)
    distances = read_json("TEST2_h6_s42_nofill.json")
    if distances:
        r = distances["resumen"]
        c1.metric("Error de distancia en test (H6)", f"{r['error_distancia_mm_media']:.2f} mm",
                  help=f"{r['n_con_distancia_predicha']} de {r['n_fragmentos_secundarios']} fragmentos secundarios "
                       f"con distancia predicha · mediana {r['error_distancia_mm_mediana']:.1f} mm · "
                       f"correlación {r['correlacion_distancia']:.2f}")
        c1.caption(f"Correlación predicción–real {r['correlacion_distancia']:.2f}; un solo fragmento concentra "
                   "la mayor parte del error medio.")
    sam = read_json("sam_baseline_test_h6.json")
    if sam:
        c2.metric("Dice por región en test: propio vs SAM zero-shot",
                  f"{sam['global_nuestro']['dice']:.3f} vs {sam['global_sam']['dice']:.3f}",
                  help=f"{sam['global_sam']['n']} regiones en {sam['cortes']} cortes (uno de cada {sam['stride']}). "
                       "SAM recibe como indicación la caja del ground truth.")


def latency_section() -> None:
    """Latencia por imagen: outputs/demo/latency.json (measure_latency.py) y las de Colab de la Semana 10."""
    import pandas as pd
    st.subheader("Latencia por imagen (384×384, lote 1)")
    rows = []
    measured = DEMO_ROOT / "latency.json"
    for name in ("latencia_gpu_L4.json", "latencia_cpu_colab.json"):
        data = read_json(name)
        if not data:
            continue
        for config, values in data["config"].items():
            rows.append({"Dispositivo": data["device"], "Configuración": config,
                         "Mediana (ms)": values["mediana_ms"], "Percentil alto (ms)": f"p95 · {values['p95_ms']}",
                         "Fuente": "Semana 10, Colab"})
    if measured.exists():
        for row in json.loads(measured.read_text()):       # filas de src/measure_latency.py
            hilos = f", {row['Hilos CPU']} hilos" if row.get("Hilos CPU") not in (None, "—") else ""
            rows.append({"Dispositivo": f"{row['Dispositivo'].lower()} ({row['Hardware']}{hilos})",
                         "Configuración": "1 modelo" + (" + TTA" if row["TTA"] == "sí" else ""),
                         "Mediana (ms)": row["Mediana (ms)"], "Percentil alto (ms)": f"p90 · {row['p90 (ms)']}",
                         "Fuente": "measure_latency.py"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Aún no hay mediciones de latencia en CPU y GPU (`src/measure_latency.py`).")
    st.caption(f"Al exportar este caso: {meta['ms_per_slice']:.0f} ms por corte ({meta['device']}, "
               f"TTA {'sí' if meta['tta'] else 'no'}, lote de 8). La inferencia en vivo de la pestaña 2 mide un "
               "corte suelto en el servicio de AWS Lambda.")


# ----------------------------------------------------------------------------- encabezado
st.title("Fracturas pélvicas en CT")
st.caption("Detección, segmentación por fragmentos y distancia de separación en milímetros · dataset PENGWIN")
st.error(NON_CLINICAL, icon="⚠️")

cases = discover_cases()
if not cases:
    st.info(f"No hay casos exportados en `{DEMO_ROOT}`. Genera uno con:\n\n"
            "`python src/demo_export.py --case 024 --checkpoint outputs/s10/h6_auggeo_rot35_canon_ema_w32_e60_s42.pth --tta`")
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
    if meta.get("canonical"):
        st.caption("Orientación canónica (sacro abajo): " + (
            "este volumen venía guardado girado 180° (cabecera LPI) y se enderezó antes de predecir; "
            "las tres vistas y el ground truth están enderezados." if meta.get("flipped")
            else "este volumen ya venía en esa orientación."))
    st.markdown(legend_html(meta["fragments_pred"]), unsafe_allow_html=True)

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
                    if live.get("model") and live["model"] != meta["model"]["checkpoint"]:
                        st.warning(f"El servicio usa {live['model']} y este caso se exportó con "
                                   f"{meta['model']['checkpoint']}.")
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

# ----------------------------------------------------------------------------- métricas
with tab4:
    st.subheader("Distancia de separación: predicción contra ground truth")
    st.caption(f"Método de la distancia: {meta.get('distance_method', 'desconocido')}")
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

    test_metrics_section()
    latency_section()
