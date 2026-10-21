"""Dashboard PENGWIN (Semana 11): tres visualizadores, métricas y latencia.

Lee las carpetas que deja ``src/demo_export.py`` en ``outputs/demo/<modelo>/<caso>/``. Para correrlo,
desde la raíz del repo:

    pip install -r requirements.txt
    python src/demo_export.py --case 024 044 --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth --tta
    streamlit run app/dashboard.py

Con ``DEMO_ROOT`` se cambia la carpeta de casos exportados. El botón de inferencia en vivo del
visualizador 2 necesita ``torch`` y el checkpoint; el resto solo lee archivos.

SUBIR UN CT. En la barra lateral, «Subir un CT» deja cargar un volumen propio (``.mha``, ``.nii``,
``.nii.gz``, ``.nrrd``) o una carpeta de cortes (DICOM, o un .mha por corte). El dashboard lo lee y lo orienta con los
metadatos (``volumen_entrada.py``), corre el mismo pipeline de ``demo_export`` (enderezado, modelo,
post-proceso 3D, distancias y mallas) y lo muestra en los tres visualizadores, sin ground truth. El
modelo corre en local si hay ``torch`` y el checkpoint; si no (Streamlit Cloud), en la API de AWS
(``INFERENCE_URL``). Los casos procesados quedan en ``UPLOAD_ROOT`` (por defecto, una carpeta temporal).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
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
RESULTS_DIR = ROOT / "outputs"                  # JSON de las evaluaciones en test
UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", Path(tempfile.gettempdir()) / "pengwin_subidos"))
MODEL_CHECKPOINT = Path(os.environ.get("MODEL_CHECKPOINT", CHECKPOINT_DIR / "h8_aumm_noise01_noema_w32_e60_s42.pth"))
MAX_UPLOAD_MB = 1024
# Vista cruda (visualizador 1) de un CT subido o procesado en vivo: por encima de estos vóxeles se calcula
# con menos resolución en el plano. Streamlit Community Cloud da ~2.7 GB por app y un .mha de 230 MB
# llegaba a 3.4 GB y tumbaba la app (sin error en el log: el sistema mata el proceso).
VISTA_MAX_VOXELES = int(os.environ.get("VISTA_MAX_VOXELES", 20_000_000))
S3_CACHE = Path(os.environ.get("S3_CACHE", Path(tempfile.gettempdir()) / "pengwin_s3"))

st.set_page_config(page_title="PENGWIN · fracturas pélvicas", page_icon="🦴", layout="wide")

# Credenciales de S3 opcionales: si están en los Secrets (usuario de solo lectura), boto3 las lee del
# entorno. Sin ellas, «Casos PENGWIN» lee el bucket a través de la API (INFERENCE_URL e INFERENCE_KEY).
for _name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "PENGWIN_BUCKET"):
    try:
        if _name not in os.environ and _name in st.secrets:
            os.environ[_name] = str(st.secrets[_name])
    except Exception:                       # no hay archivo de secretos
        break


def discover_cases() -> dict[str, Path]:
    cases = {}
    for meta in sorted(DEMO_ROOT.glob("*/*/meta.json")):
        cases[f"{meta.parent.parent.name} · caso {meta.parent.name}"] = meta.parent
    return cases


def free_memory() -> None:
    """Suelta los casos que la app tiene en memoria antes de procesar un CT nuevo.

    Streamlit Cloud tiene ~2.7 GB por app y la memoria la comparten todas las sesiones: dos casos ya
    abiertos (volúmenes y mallas) más un CT grande en proceso la pasaban. El caso se vuelve a leer del
    disco cuando se lo pide otra vez (unos segundos).
    """
    import gc
    load_case.clear()
    gc.collect()


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
    steps = folder / "pasos.npz"
    if steps.exists():
        with np.load(steps) as arrays:
            data["pasos"] = {key: arrays[key] for key in arrays.files}
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


def remote_predict_slice(url: str, key: str | None, image_u8: np.ndarray, timeout: float = 120.0,
                         destino: str | None = None) -> dict:
    """Manda un corte al servicio de inferencia (api/server.py) y devuelve lo mismo que predict_slice."""
    import base64
    import io
    import urllib.error
    import urllib.request
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(image_u8, dtype=np.uint8), allow_pickle=False)
    request = urllib.request.Request(url.rstrip("/") + "/predict_slice", data=buffer.getvalue(), method="POST",
                                     headers={"Content-Type": "application/octet-stream", "X-API-Key": key or "",
                                              **({"X-Destino": destino} if destino else {})})
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
            "device": payload["device"], "round_trip_ms": round_trip, "model": payload.get("model"),
            "precision": payload.get("precision", "fp32")}


@st.cache_resource(show_spinner="Cargando el modelo…")
def load_live_model(checkpoint: str, tta: bool, precision: str = "fp32"):
    from predict_volume import load_pipeline
    return load_pipeline(checkpoint, tta=tta, precision=precision)


def legend_html(fragments: list[dict]) -> str:
    chips = "".join(f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
                    f'<span style="width:14px;height:14px;background:{color};border-radius:3px;'
                    f'margin-right:6px;display:inline-block;"></span>{label}</span>'
                    for color, label in legend_items(fragments))
    return f'<div style="margin:4px 0 10px 0;">{chips}<span style="opacity:.7">· los secundarios son tintes más claros con contorno blanco</span></div>'


def fragments_dataframe(rows: list[dict]):
    import pandas as pd
    return pd.DataFrame([{"Hueso": REGION_LABELS[r["region"]], "Fragmento": local_name(r["id"]),
                          "Distancia al principal": "—" if r["is_main"] else distance_text(r["distance_mm"])
                          + ("  ⚠ metal" if r.get("metal") else ""),
                          "Volumen (mm³)": round(r["volume_mm3"]), "Cortes": f"{r['z_range'][0]}–{r['z_range'][1]}"}
                         for r in rows])


def read_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


TEST_ROWS = (("F1 de clasificación", "classification_f1_macro", 0.85),
             ("AUC de clasificación", "classification_auc_macro", 0.85),
             ("IoU medio de detección", "detection_iou_mean", 0.65),
             ("mAP50", "detection_map50", 0.65),
             ("mAP50:95", "detection_map50_95", 0.40),
             ("Dice de segmentación semántica", "segmentation_dice_macro", 0.85),
             ("IoU de segmentación semántica", "segmentation_iou_macro", 0.70),
             ("Dice por fragmento", "instance_dice_matched", 0.85),
             ("IoU por fragmento", "instance_iou_matched", 0.70))


TEST_EVALUATIONS = (("1.ª · D5", "s10/TEST_d5_s42_tta.json"),
                    ("2.ª · H6", "s10/TEST2_h6_s42_tta.json"),
                    ("3.ª · H7", "h7/test_tercera_h7_auggeo_rot35_canon_aumm_noema_w32_e60_s42_sin_enderezar.json"),
                    ("4.ª · H8", "h8/test/test_h8.json"))


def test_metrics_section() -> None:
    """Las evaluaciones en test del proyecto, leídas de outputs/ (no se recalcula nada)."""
    import pandas as pd
    found = [(label, read_json(name)) for label, name in TEST_EVALUATIONS]
    found = [(label, data["subsets"]["todo"]) for label, data in found if data]
    final = dict(found).get("4.ª · H8")
    if not final:
        return
    slices = f"{final['n_slices']:,}".replace(",", " ")
    st.subheader(f"Resultados en test ({final['n_cases']} casos, {slices} cortes)")
    rows = []
    for name, key, goal in TEST_ROWS:
        row = {"Métrica": name, "Objetivo": f"{goal:.2f}"}
        row.update({label: f"{values[key]:.3f}" for label, values in found})
        row["Cumple (H8)"] = "sí" if final[key] >= goal else "no"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("**Cómo leer las columnas.** H8 es el modelo de esta demo: sin EMA, con el giro del aumento en "
               "milímetros y con aumento de ruido gaussiano; se eligió solo con validación y se evaluó en test una "
               "vez, con la configuración escrita de antemano. La primera evaluación (D5) se hizo con la "
               "configuración congelada antes de ver test y es la única cifra limpia: no cumplía el Dice por "
               "fragmento (0.845). Las siguientes se hicieron después de analizar los fallos de esa primera "
               "evaluación, que miró dos casos de test (012 y 039).")
    c1, c2, c3 = st.columns(3)
    surface = read_json("h8/test/superficie_test_h8.json")
    if surface:
        r = surface["resumen"]["todos"]
        c1.metric("HD95 / ASSD por fragmento en test (H8)",
                  f"{r['hd95_mm_mediana']:.1f} / {r['assd_mm_mediana']:.2f} mm",
                  help=f"Medianas sobre {r['n_detectados']} de {r['n_fragmentos']} fragmentos reales detectados "
                       f"(cobertura {r['cobertura']:.1%}) · medias {r['hd95_mm_media']:.1f} / "
                       f"{r['assd_mm_media']:.2f} mm")
    distances, who = read_json("h8/test/distancias_test_h8.json"), "H8"
    if not distances:
        distances, who = read_json("s10/TEST2_h6_s42_nofill.json"), "H6"
    if distances:
        r = distances["resumen"]
        c2.metric(f"Error de distancia en test ({who})", f"{r['error_distancia_mm_media']:.2f} mm",
                  help=f"{r['n_con_distancia_predicha']} de {r['n_fragmentos_secundarios']} fragmentos secundarios "
                       f"con distancia predicha · mediana {r['error_distancia_mm_mediana']:.1f} mm · "
                       f"correlación {r['correlacion_distancia']:.2f}")
    sam, who = read_json("h8/test/sam_baseline_test_h8.json"), "H8"
    if not sam:
        sam, who = read_json("s10/sam_baseline_test_h6.json"), "H6"
    if sam:
        c3.metric(f"Dice por región en test: {who} vs SAM zero-shot",
                  f"{sam['global_nuestro']['dice']:.3f} vs {sam['global_sam']['dice']:.3f}",
                  help=f"{sam['global_sam']['n']} regiones en {sam['cortes']} cortes (uno de cada {sam['stride']}). "
                       f"SAM recibe como indicación la caja que predice {who}; no forma parte del pipeline.")


def latency_section() -> None:
    """Latencia por imagen: outputs/demo/latency.json (measure_latency.py) y las de Colab de la Semana 10."""
    import pandas as pd
    st.subheader("Latencia por imagen (384×384, lote 1)")
    rows = []
    measured = DEMO_ROOT / "latency.json"
    for name in ("s10/latencia_gpu_L4.json", "s10/latencia_cpu_colab.json"):
        data = read_json(name)
        if not data:
            continue
        for config, values in data["config"].items():
            rows.append({"Dispositivo": data["device"], "Configuración": config,
                         "Mediana (ms)": values["mediana_ms"], "Percentil alto (ms)": f"p95 · {values['p95_ms']}",
                         "Fuente": "Semana 10, Colab (H6, misma arquitectura)"})
    if measured.exists():
        for row in json.loads(measured.read_text(encoding="utf-8")):       # filas de src/measure_latency.py
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


@st.cache_resource(show_spinner="Cargando el modelo…")
def local_predictor(checkpoint: str, tta: bool, precision: str = "fp32"):
    from demo_export import predictor_local
    return predictor_local(checkpoint, tta=tta, precision=precision)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


PRECISION_LABELS = {"auto": "Automática: int8 en CPU, fp16 en GPU", "fp32": "H8 normal (fp32)",
                    "int8": "H8 cuantizado (int8, para CPU)", "fp16": "H8 fp16 (para GPU)"}


@st.cache_data(ttl=60, show_spinner=False)
def gpu_status(url: str) -> str | None:
    """Estado del endpoint de GPU según la API («InService» si está encendido); None si no hay o no responde."""
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=10) as response:
            return json.loads(response.read()).get("gpu")
    except Exception:
        return None


def processing_targets() -> dict[str, dict]:
    """Dónde puede correr el modelo: este equipo (torch + checkpoint), la API en CPU y, si está, en GPU."""
    targets = {}
    if torch_available() and MODEL_CHECKPOINT.exists():
        targets["Este equipo"] = {"kind": "local", "slug": "local"}
    if setting("INFERENCE_URL"):
        targets["AWS · CPU (Lambda)"] = {"kind": "remote", "slug": "aws-cpu", "url": setting("INFERENCE_URL"),
                                         "key": setting("INFERENCE_KEY")}
    if setting("INFERENCE_URL") and gpu_status(setting("INFERENCE_URL")) == "InService":
        # El endpoint de SageMaker existe y está encendido: la Lambda reenvía los pedidos con X-Destino: gpu.
        targets["AWS · GPU (SageMaker)"] = {"kind": "remote", "slug": "aws-gpu", "url": setting("INFERENCE_URL"),
                                            "key": setting("INFERENCE_KEY"), "destino": "gpu"}
    if setting("INFERENCE_URL_GPU"):
        targets["AWS · GPU"] = {"kind": "remote", "slug": "aws-gpu", "url": setting("INFERENCE_URL_GPU"),
                                "key": setting("INFERENCE_KEY_GPU") or setting("INFERENCE_KEY")}
    return targets


@st.cache_data(ttl=30, show_spinner=False)
def gpu_details(url: str, key: str) -> dict | None:
    from inferencia_remota import control_gpu
    try:
        return control_gpu(url, key, "estado")
    except Exception:
        return None


def falta(iso: str | None) -> str:
    """«3 h 12 min» hasta la hora UTC ``iso`` (2026-10-17T22:00:00Z)."""
    if not iso:
        return ""
    from datetime import datetime, timezone
    minutos = max(0, int((datetime.fromisoformat(iso.replace("Z", "+00:00"))
                          - datetime.now(timezone.utc)).total_seconds() // 60))
    return f"{minutos // 60} h {minutos % 60:02d} min"


GPU_TEXTO = {"InService": "🟢 encendida", "Creating": "🟡 encendiendo (~8 min)", "Updating": "🟡 actualizando",
             "Deleting": "🟠 apagándose", "Failed": "🔴 falló al encender", None: "⚪ apagada"}


def gpu_panel() -> None:
    """GPU en SageMaker bajo demanda: estado, encender (con apagado automático) y apagar.

    Cada hora encendida cuesta ~0.74 USD; por eso encender y apagar piden la clave de administración
    del equipo (``GPU_ADMIN_KEY`` en la Lambda) y la GPU se apaga sola a las 4 horas.
    """
    url, key = setting("INFERENCE_URL"), setting("INFERENCE_KEY")
    if not url:
        return
    datos = gpu_details(url, key or "")
    if not datos or not datos.get("endpoint"):
        return                                   # la API no tiene GPU configurada
    with st.sidebar.expander(f"GPU (SageMaker) · {GPU_TEXTO.get(datos['estado'], datos['estado'])}"):
        st.caption(f"NVIDIA T4 (`ml.g4dn.xlarge`, ~0.74 USD/h mientras esté encendida). Encenderla tarda unos "
                   f"8 minutos; se apaga sola a las {datos.get('horas_apagado', 4):g} h.")
        if datos.get("apagado_programado"):
            st.write(f"Se apaga sola en **{falta(datos['apagado_programado'])}** "
                     f"({datos['apagado_programado'].replace('T', ' ').replace('Z', ' UTC')}).")
        clave = st.text_input("Clave de administración", type="password", key="gpu_admin",
                              help="Solo el equipo: evita que cualquier visitante de la app encienda la GPU.")
        encender, apagar = st.columns(2)
        accion = None
        if encender.button("Encender (4 h)" if datos["estado"] is None else "Extender a 4 h",
                           disabled=datos["estado"] == "Deleting", use_container_width=True):
            accion = "encender"
        if apagar.button("Apagar", disabled=datos["estado"] in (None, "Deleting"), use_container_width=True):
            accion = "apagar"
        if accion:
            from inferencia_remota import control_gpu
            try:
                control_gpu(url, key, accion, clave)
            except RuntimeError as error:
                st.error(f"No se pudo {accion}: {error}")
            else:
                gpu_details.clear()
                gpu_status.clear()
                st.rerun()
        if datos["estado"] == "Creating" and st.button("Actualizar estado", use_container_width=True):
            gpu_details.clear()
            gpu_status.clear()
            st.rerun()


def processing_settings() -> dict | None:
    """Bloque «Procesamiento» de la barra lateral: dónde corre el modelo y con qué precisión."""
    targets = processing_targets()
    if not targets:
        return None
    with st.sidebar:
        st.subheader("Procesamiento")
        name = st.selectbox("Dónde corre el modelo", list(targets), key="target",
                            help="Para la inferencia en vivo, «Procesar ahora» y los CT subidos. Los casos ya "
                                 "precalculados no se vuelven a procesar.")
        target = dict(targets[name], name=name)
        if target["kind"] == "local":
            import torch
            gpu = torch.cuda.is_available()
            target["precision"] = st.selectbox(
                "Precisión", list(PRECISION_LABELS), format_func=PRECISION_LABELS.get, key="precision",
                help="El mismo H8 con el mismo Dice (docs/cuantizacion.md); cambia la velocidad. int8 solo corre "
                     "en CPU y fp16 rinde en GPU." + ("" if gpu else " Este equipo no tiene GPU: fp16 será lento."))
            if target["precision"] == "fp16" and not gpu:
                st.caption("Sin GPU, fp16 es más lento que fp32: sirve solo para comparar.")
        else:
            target["precision"] = "servicio"
            st.caption("La precisión la fija el servicio (`PRECISION`; con `auto`, int8 en CPU y fp16 en GPU) y "
                       "se informa al procesar.")
    gpu_panel()
    return target


def choose_predictor(tta: bool, target: dict | None):
    """Devuelve (info, predecir, dónde) según el destino elegido en «Procesamiento»."""
    if target is None:
        raise RuntimeError("No hay dónde correr el modelo: falta torch con el checkpoint en "
                           f"`{MODEL_CHECKPOINT.relative_to(ROOT)}` o el servicio de inferencia (`INFERENCE_URL`).")
    if target["kind"] == "local":
        info, predecir = local_predictor(str(MODEL_CHECKPOINT), tta, target["precision"])
        return info, predecir, f"este equipo ({info['device']}, {info['precision']})"
    from inferencia_remota import predictor_remoto
    info, predecir = predictor_remoto(target["url"], target["key"], destino=target.get("destino"))
    return info, predecir, f"{target['name']} ({info['precision']})"


def max_upload_voxels() -> int | None:
    """Tope de vóxeles para procesar un CT en el dashboard publicado (sin torch: Streamlit Cloud, ~2.7 GB).

    El caso más grande de PENGWIN (001) tiene 105 millones de vóxeles y cabe con margen. En local no hay
    tope salvo que se fije ``MAX_VOXELES_SUBIDA``."""
    value = setting("MAX_VOXELES_SUBIDA")
    if value:
        return int(value)
    return None if torch_available() else 120_000_000


def check_volume_size(voxels: int) -> None:
    limit = max_upload_voxels()
    if limit and voxels > limit:
        raise RuntimeError(f"El volumen tiene {voxels / 1e6:.0f} millones de vóxeles y el dashboard publicado procesa "
                           f"hasta {limit / 1e6:.0f} millones (memoria limitada). Úsalo en el dashboard local "
                           "(`streamlit run app/dashboard.py`), que no tiene ese tope.")


def upload_page() -> Path | None:
    """Carga de un CT propio. Devuelve la carpeta del caso procesado, o None si todavía no hay uno."""
    st.subheader("Subir un CT propio")
    st.caption("Un volumen en un solo archivo (.mha, .nii, .nii.gz, .nrrd; un .mhd con su .raw) o una carpeta "
               "de cortes: los DICOM de una serie o un .mha por corte. Se orienta con los metadatos, se endereza el giro del "
               "paciente si es claro, y pasa por el mismo pipeline que los casos de ejemplo. No hay ground "
               "truth, así que no se calculan métricas. Una carpeta de PNG o JPG no sirve: no guarda las "
               "unidades Hounsfield ni el tamaño del vóxel.")
    modo = st.radio("Qué vas a subir", ["Archivo de volumen", "Carpeta de cortes (DICOM o .mha)"], horizontal=True)
    if modo == "Archivo de volumen":
        files = st.file_uploader("Volumen", type=["mha", "mhd", "raw", "zraw", "nii", "gz", "nrrd", "nhdr"],
                                 accept_multiple_files=True, max_upload_size=MAX_UPLOAD_MB,
                                 help="Un .mhd o .nhdr se sube junto con su archivo de datos.")
    else:
        files = st.file_uploader("Carpeta con los cortes", accept_multiple_files="directory",
                                 max_upload_size=MAX_UPLOAD_MB,
                                 help="Una serie DICOM, o un .mha (.nii, .nrrd) por corte. Los .mha se ordenan por "
                                      "su posición en el paciente si la traen.")
    grosor = None
    if modo == "Archivo de volumen":
        grosor = st.number_input("Corregir el grosor entre cortes (mm), opcional", min_value=0.0,
                                 max_value=20.0, value=0.0, step=0.1, format="%.2f",
                                 help="Solo si la cabecera está mal (por ejemplo, dice 100 mm entre cortes). "
                                      "Reemplaza la distancia entre cortes del archivo; déjalo en 0 para usar "
                                      "la de la cabecera.") or None
    else:
        grosor = st.number_input("Grosor entre cortes (mm), solo para cortes .mha en 2D", min_value=0.0,
                                 max_value=20.0, value=0.0, step=0.1, format="%.2f",
                                 help="Un .mha 2D no guarda su posición en el paciente, así que no se sabe la "
                                      "distancia entre cortes. Si los cortes la traen (o es una serie DICOM), "
                                      "déjalo en 0 y se usa la de los archivos.") or None
    c1, c2 = st.columns(2)
    tta = c1.checkbox("TTA de espejo", False, help="Predice también la imagen en espejo y promedia: un poco "
                      "más preciso y el doble de tiempo. Solo aplica con el modelo local.")
    enderezar = c2.checkbox("Enderezar el giro axial del paciente", True,
                            help="Estima el giro con la simetría de la pelvis y lo corrige si es de 10° o más.")
    if not st.button("Procesar", type="primary", disabled=not files):
        return None
    from demo_export import exportar_volumen
    from volumen_entrada import huella, leer_subida
    archivos = [(f.name, f.getbuffer()) for f in files]     # sin copiar los bytes subidos (memoria)
    free_memory()
    with st.status("Procesando el CT…", expanded=True) as status:
        try:
            info, predecir, where = choose_predictor(tta, PROCESSING)
            out_dir = UPLOAD_ROOT / Path(info["checkpoint"]).stem / (
                f"{huella(archivos)}-{PROCESSING['slug']}-{info.get('precision', 'fp32')}"
                f"{'-tta' if info.get('tta') else ''}{'' if enderezar else '-sinenderezar'}{f'-g{grosor:g}' if grosor else ''}")
            if (out_dir / "meta.json").exists():
                status.update(label="Este CT ya se había procesado", state="complete")
                return out_dir
            status.write(f"Modelo: {info['checkpoint']} · {where}")
            volume = leer_subida(archivos, orientar=bool(info.get("canonical")), grosor_mm=grosor)
            del archivos
            check_volume_size(volume.image.size)
            status.write(f"Leído: {volume.fuente} · {volume.image.shape[0]} cortes de "
                         f"{volume.image.shape[1]}×{volume.image.shape[2]} · vóxel "
                         f"{'×'.join(f'{v:.2f}' for v in volume.spacing_xyz)} mm · orientación: "
                         f"{volume.orientacion['accion']}")
            started = time.perf_counter()
            exportar_volumen(volume.image, volume.spacing_xyz, out_dir, info, predecir, nombre=volume.nombre,
                             enderezar=enderezar, log=status.write, pasos=volume.pasos,
                             max_voxeles_vista=VISTA_MAX_VOXELES,
                             extra_meta={"subido": {"fuente": volume.fuente, "orientacion": volume.orientacion,
                                                    "donde": where}})
            status.update(label=f"Listo en {time.perf_counter() - started:.0f} s", state="complete")
        except Exception as error:
            status.update(label="No se pudo procesar", state="error")
            st.error(str(error))
            return None
    return out_dir


@st.cache_data(ttl=300, show_spinner="Consultando los casos en S3…")
def s3_cases(model: str) -> list[dict]:
    import almacen_s3
    return almacen_s3.listar_casos(model)


def pengwin_page() -> Path | None:
    """Los 100 casos de PENGWIN desde S3: el precalculado o procesarlo en este momento."""
    import almacen_s3
    almacen_s3.configurar_api(setting("INFERENCE_URL"), setting("INFERENCE_KEY"))
    model = MODEL_CHECKPOINT.stem
    try:
        cases = s3_cases(model)
    except Exception as error:
        st.error(f"No pude leer el bucket `{almacen_s3.BUCKET}`: {error}. Hace falta un perfil de AWS con acceso "
                 "(por ejemplo `AWS_PROFILE=almia` en local) o la API de inferencia (`INFERENCE_URL` e "
                 "`INFERENCE_KEY`), que lee el bucket por su cuenta.")
        return None
    if not cases:
        st.info(f"El bucket `{almacen_s3.BUCKET}` todavía no tiene casos.")
        return None
    ready = sum(c["precalculado"] for c in cases)
    with st.sidebar:
        case = st.selectbox(f"Caso ({len(cases)} de PENGWIN · {ready} precalculados)", cases, key="pengwin_case",
                            format_func=lambda c: f"{c['caso']} · {c['split']}"
                            + (" · girado 180°" if c["girado"] else "")
                            + (" · precalculado" if c["precalculado"] else ""))
        options = (["Precalculado", "Procesar ahora"] if case["precalculado"] and case["crudo"]
                   else ["Precalculado"] if case["precalculado"] else ["Procesar ahora"])
        mode = st.radio("Resultado", options, horizontal=True, key="pengwin_mode",
                        help="Precalculado: se descarga de S3 en segundos. Procesar ahora: descarga el .mha "
                             "original y corre el pipeline completo en vivo (unos minutos).")
    case_id = case["caso"]
    if mode == "Precalculado":
        folder = S3_CACHE / "precalculados" / model / case_id
        if not (folder / "meta.json").exists():
            with st.spinner(f"Descargando el caso {case_id} de S3…"):
                almacen_s3.descargar_exportado(model, case_id, folder)
        return folder

    st.subheader(f"Caso {case_id} de PENGWIN · procesar ahora")
    st.caption(f"Split {case['split']}" + (" · la cabecera dice que está guardado girado 180°" if case["girado"]
                                            else "") + ". Se descarga el .mha original de S3 (con su ground "
               "truth) y pasa por todo el pipeline: orientación con los metadatos, giro axial, modelo, "
               "post-proceso 3D, distancias y mallas.")
    tta = st.checkbox("TTA de espejo", False, key="pengwin_tta",
                      help="Más preciso y el doble de tiempo. Solo aplica con el modelo local.")
    target = PROCESSING or {"slug": "ninguno", "precision": "fp32"}
    folder = S3_CACHE / "en_vivo" / model / f"{case_id}-{target['slug']}-{target['precision']}{'-tta' if tta else ''}"
    if (folder / "meta.json").exists():
        return folder
    if not st.button(f"Procesar el caso {case_id}", type="primary"):
        return None
    from demo_export import exportar_volumen, preparar_caso
    free_memory()
    with st.status(f"Procesando el caso {case_id}…", expanded=True) as status:
        try:
            info, predecir, where = choose_predictor(tta, PROCESSING)
            status.write(f"Modelo: {info['checkpoint']} · {where}")
            started = time.perf_counter()
            with tempfile.TemporaryDirectory() as tmp:
                status.write("Descargando el .mha original de S3…")
                image_path, label_path = almacen_s3.descargar_crudo(case_id, Path(tmp))
                image, label, spacing_xyz, flipped, steps = preparar_caso(
                    image_path, label_path, case_id, bool(info.get("canonical")), log=status.write)
            exportar_volumen(image, spacing_xyz, folder, info, predecir, label=label, nombre=case_id,
                             flipped=flipped, pasos=steps, log=status.write, max_voxeles_vista=VISTA_MAX_VOXELES,
                             extra_meta={"en_vivo": {"donde": where}})
            status.update(label=f"Listo en {time.perf_counter() - started:.0f} s", state="complete")
        except Exception as error:
            status.update(label="No se pudo procesar", state="error")
            st.error(str(error))
            return None
    st.rerun()


def steps_tab(meta: dict, arrays: dict | None) -> None:
    st.subheader("Preprocesamiento paso a paso")
    if not meta.get("pasos") or not arrays:
        st.info("Este caso se exportó antes de que existiera esta vista. Vuelve a exportarlo (o elige un caso "
                "de PENGWIN o sube un CT) para verla.")
        return
    st.caption("Qué le pasa al volumen antes de entrar a la red. A la izquierda, el corte axial con más hueso "
               "(el mismo en todas las etapas); a la derecha, la proyección de máxima intensidad (MIP) axial. "
               "Las etapas 1 a 3 se ven con la ventana ósea y la proporción real del vóxel; la 4, tal cual entra "
               "al modelo.")
    for number, step in enumerate(meta["pasos"], 1):
        st.markdown(f"**{number}. {step['titulo']}**")
        text, cut, mip = st.columns([2, 2, 2])
        text.write(step["texto"])
        if step.get("forma"):
            text.caption(f"{step['forma'][0]} cortes de {step['forma'][1]}×{step['forma'][2]} · corte mostrado: "
                         f"{(step.get('z') or 0) + 1}")
        cut.image(arrays[f"{step['clave']}_corte"], caption="Corte axial", use_container_width=True)
        mip.image(arrays[f"{step['clave']}_mip"], caption="MIP axial", use_container_width=True)
        st.divider()


# ----------------------------------------------------------------------------- encabezado
st.title("Fracturas pélvicas en CT")
st.caption("Detección, segmentación por fragmentos y distancia de separación en milímetros · dataset PENGWIN")
st.error(NON_CLINICAL, icon="⚠️")

with st.sidebar:
    st.header("Caso")
    source = st.radio("Origen", ["Casos de ejemplo", "Casos PENGWIN (100)", "Subir un CT"], key="source")
PROCESSING = processing_settings()

if source == "Casos PENGWIN (100)":
    case_path = pengwin_page()
    if case_path is None:
        st.stop()
elif source == "Subir un CT":
    uploaded = st.session_state.get("uploaded_case")
    if uploaded and Path(uploaded, "meta.json").exists():
        with st.sidebar:
            if st.button("Subir otro CT"):
                st.session_state.pop("uploaded_case")
                st.rerun()
        case_path = Path(uploaded)
    else:
        case_path = upload_page()
        if case_path is None:
            st.stop()
        st.session_state["uploaded_case"] = str(case_path)
        st.rerun()
else:
    cases = discover_cases()
    if not cases:
        st.info(f"No hay casos exportados en `{DEMO_ROOT}`. Genera uno con:\n\n"
                "`python src/demo_export.py --case 024 --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth --tta`")
        st.stop()
    with st.sidebar:
        choice = st.selectbox("Modelo y caso exportado", list(cases))
    case_path = cases[choice]

with st.sidebar:
    data = load_case(str(case_path))
    meta = data["meta"]
    subido = meta.get("subido")
    if subido:
        st.caption(f"CT subido: {subido['fuente']} · orientación con los metadatos: "
                   f"{subido['orientacion']['accion']} · inferencia: {subido['donde']}")
    st.caption(f"{meta['n_slices']} cortes · {meta['size']}×{meta['size']} px · spacing del header "
               f"{meta['spacing_native_xyz'][0]:.2f}×{meta['spacing_native_xyz'][1]:.2f}×"
               f"{meta['spacing_native_xyz'][2]:.2f} mm")
    st.caption(f"Modelo {meta['model']['checkpoint']} · {meta['model']['parameters'] / 1e6:.2f} M parámetros "
               f"· precisión {meta['model'].get('precision', 'fp32')} · TTA {'sí' if meta['tta'] else 'no'}")
    if meta.get("canonical") and not subido:
        st.caption("Orientación canónica (sacro abajo): " + (
            "este volumen venía guardado girado 180° (cabecera LPI) y se enderezó antes de predecir; "
            "las tres vistas y el ground truth están enderezados." if meta.get("flipped")
            else "este volumen ya venía en esa orientación."))
    giro = meta.get("enderezado")
    if giro:
        st.caption(f"Giro axial del paciente: {giro['angulo']:+.1f}° estimado; " + (
            "se enderezó antes de predecir (las tres vistas y el ground truth están enderezados)."
            if giro["corregido"] else f"no se corrigió (solo se corrigen giros claros de {giro['giro_minimo']:.0f}° o más)."))
    for aviso in meta.get("avisos") or []:
        st.warning(f"**{aviso['titulo']}.** {aviso['texto']}", icon="⚠️")
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

tab0, tab1, tab2, tab3, tab4 = st.tabs(["Preprocesamiento", "1 · Volumen crudo", "2 · Corte por corte",
                                        "3 · Reconstrucción 3D", "Métricas y latencia"])

with tab0:
    steps_tab(meta, data.get("pasos"))

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
            verts, faces = raw["bone_verts_zyx"], raw["bone_faces"]
            if "bone_verts_zyx_pulido" in raw and "bone_faces_pulido" in raw:
                version = st.radio("Hueso", ["Pulido", "Crudo"], horizontal=True,
                                   help="Pulido: además del umbral de 250 HU recupera el hueso tenue "
                                        "(150-250 HU conectado al esqueleto) y rellena las cavidades. "
                                        "Crudo: solo el umbral y la limpieza, como la MIP.")
                if version == "Pulido":
                    verts, faces = raw["bone_verts_zyx_pulido"], raw["bone_faces_pulido"]
            st.plotly_chart(bone_figure(verts, faces), use_container_width=True)
    st.caption("El pulido 3D es solo visual; el modelo, las métricas y las distancias usan los datos sin modificar.")

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
        checkpoint = CHECKPOINT_DIR / meta["model"]["checkpoint"]
        if PROCESSING is None or (PROCESSING["kind"] == "local" and not checkpoint.exists()):
            st.info("La inferencia en vivo no está configurada en esta instalación: falta el servicio de "
                    "inferencia (`INFERENCE_URL`) o el checkpoint en `outputs/s10/`.")
        elif st.button(f"Inferir el corte seleccionado · {PROCESSING['name']}"):
            try:
                if PROCESSING["kind"] == "remote":
                    live = remote_predict_slice(PROCESSING["url"], PROCESSING["key"], data["images"][z],
                                                destino=PROCESSING.get("destino"))
                    where = (f"{live['device']} en {PROCESSING['name']} · precisión {live.get('precision', 'fp32')} · "
                             f"ida y vuelta {live['round_trip_ms']:.0f} ms")
                    if live.get("model") and live["model"] != meta["model"]["checkpoint"]:
                        st.warning(f"El servicio usa {live['model']} y este caso se exportó con "
                                   f"{meta['model']['checkpoint']}.")
                else:
                    from predict_volume import predict_slice
                    model, device, info = load_live_model(str(checkpoint), False, PROCESSING["precision"])
                    predict_slice(model, device, data["images"][z])     # pasada de calentamiento
                    live = predict_slice(model, device, data["images"][z])
                    where = f"{device} de este equipo · precisión {info['precision']}"
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
    st.caption("El pulido 3D es solo visual; el modelo, las métricas y las distancias usan los datos sin modificar.")

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
