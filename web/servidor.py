"""API de la página web de AWS (``web/frontend``): lo mismo que hace el dashboard de Streamlit, en el servidor.

El dashboard de Streamlit Cloud procesa los CT en su propio proceso, con ~2.7 GB de memoria: un .mha grande
lo puede tumbar. Esta API corre en una Lambda propia (``pelvis-web``, 10 GB de memoria, 15 minutos) y deja
al navegador solo el dibujo. El dashboard de Streamlit y la Lambda ``pelvis-api`` no cambian.

Flujo:

- ``GET /api/config``: modelo, dónde puede correr (CPU de esta Lambda; GPU de SageMaker si está encendida) y
  el estado de la GPU.
- ``GET /api/casos``: casos de ejemplo y los 100 de PENGWIN (bucket ``PENGWIN_BUCKET``).
- ``POST /api/abrir``: abre un caso de PENGWIN, precalculado o «procesar ahora». Si su paquete web ya
  existe, responde con él; si no, crea un **trabajo**.
- ``POST /api/subidas`` → formularios firmados para subir el CT directo a S3 (sin pasar por la Lambda, que
  acepta hasta 6 MB por pedido); luego ``POST /api/subidas/{id}/procesar`` crea el trabajo.
- ``GET /api/trabajos/{id}``: estado y bitácora del trabajo, para la barra de progreso.
- ``POST /api/corte``: inferencia en vivo de un corte (visualizador 2), en esta CPU o en la GPU.
- ``GET /api/gpu`` y ``POST /api/gpu/{encender|apagar}``: la GPU bajo demanda, a través de ``pelvis-api``
  (que tiene los permisos). Encender y apagar piden la clave de administración del equipo.

TRABAJOS. Un trabajo corre en otra invocación de esta misma Lambda, asíncrona: el Lambda Web Adapter entrega
el evento a ``POST /events``. Ahí se corre el pipeline de ``demo_export`` completo (el modelo corre en esta
misma Lambda, o en la GPU si se eligió) y ``paquete_web.empaquetar`` deja el resultado en
``WEB_BUCKET/resultados/...``, que CloudFront sirve al navegador. El estado vive en ``trabajos/<id>.json``.

Sin ``WEB_BUCKET`` (en el Mac) todo queda en ``WEB_DATA_DIR`` y los trabajos corren en un hilo:

    AWS_PROFILE=almia WEB_DATA_DIR=/tmp/pengwin_web uvicorn web.servidor:app --port 8080
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "outputs" / "s10" / "h8_aumm_noise01_noema_w32_e60_s42.pth"))
MODELO = Path(MODEL_PATH).stem
PRECISION = os.environ.get("PRECISION", "auto")
WEB_BUCKET = os.environ.get("WEB_BUCKET", "")
WEB_REGION = os.environ.get("WEB_REGION", os.environ.get("AWS_REGION", "us-east-2"))
WEB_DATA_DIR = Path(os.environ.get("WEB_DATA_DIR", Path(tempfile.gettempdir()) / "pengwin_web"))
INFERENCE_URL = os.environ.get("INFERENCE_URL", "")
INFERENCE_KEY = os.environ.get("INFERENCE_KEY", "")
MAX_SUBIDA_MB = int(os.environ.get("MAX_SUBIDA_MB", "1024"))
MAX_ARCHIVOS = 3000
# Vista cruda (visualizador 1) de un CT grande: se calcula con menos resolución en el plano, como en el
# dashboard. Es solo visual y evita que el caso más grande de PENGWIN (105 M vóxeles) pase de 10 GB.
VISTA_MAX_VOXELES = int(os.environ.get("VISTA_MAX_VOXELES", 20_000_000))
TAMANO = 384

app = FastAPI(title="PENGWIN web", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------------- almacenamiento
class AlmacenLocal:
    """Carpeta local con la misma interfaz que S3 (desarrollo en el Mac y pruebas)."""

    def __init__(self, raiz: Path) -> None:
        self.raiz = Path(raiz)

    def leer(self, clave: str) -> bytes | None:
        ruta = self.raiz / clave
        return ruta.read_bytes() if ruta.exists() else None

    def escribir(self, clave: str, datos: bytes, tipo: str | None = None) -> None:
        ruta = self.raiz / clave
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = ruta.with_name(ruta.name + ".tmp")
        tmp.write_bytes(datos)
        tmp.replace(ruta)

    def existe(self, clave: str) -> bool:
        return (self.raiz / clave).exists()

    def subcarpetas(self, prefijo: str) -> list[str]:
        carpeta = self.raiz / prefijo
        return sorted(p.name for p in carpeta.iterdir() if p.is_dir()) if carpeta.exists() else []

    def descargar(self, prefijo: str, destino: Path) -> list[Path]:
        carpeta = self.raiz / prefijo
        rutas = []
        for ruta in sorted(carpeta.iterdir()) if carpeta.exists() else []:
            shutil.copyfile(ruta, destino / ruta.name)
            rutas.append(destino / ruta.name)
        return rutas

    def formulario(self, clave: str, max_bytes: int) -> dict:
        return {"url": "/api/dev/subir", "fields": {"key": clave}}


class AlmacenS3:
    def __init__(self, bucket: str, region: str) -> None:
        import boto3
        from botocore.config import Config
        self.bucket = bucket
        self.s3 = boto3.client("s3", region_name=region, endpoint_url=f"https://s3.{region}.amazonaws.com",
                               config=Config(signature_version="s3v4", max_pool_connections=32))

    def leer(self, clave: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=clave)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def escribir(self, clave: str, datos: bytes, tipo: str | None = None) -> None:
        extra = {"ContentType": tipo} if tipo else {}
        if clave.startswith("trabajos/"):
            extra["CacheControl"] = "no-store"
        self.s3.put_object(Bucket=self.bucket, Key=clave, Body=datos, **extra)

    def existe(self, clave: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=clave)
            return True
        except Exception:
            return False

    def subcarpetas(self, prefijo: str) -> list[str]:
        prefijo = prefijo.rstrip("/") + "/"
        nombres, token = [], None
        while True:
            extra = {"ContinuationToken": token} if token else {}
            r = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefijo, Delimiter="/", **extra)
            nombres += [p["Prefix"][len(prefijo):].strip("/") for p in r.get("CommonPrefixes", [])]
            if not r.get("IsTruncated"):
                return sorted(nombres)
            token = r["NextContinuationToken"]

    def descargar(self, prefijo: str, destino: Path) -> list[Path]:
        claves, token = [], None
        while True:
            extra = {"ContinuationToken": token} if token else {}
            r = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefijo.rstrip("/") + "/", **extra)
            claves += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                break
            token = r["NextContinuationToken"]
        rutas = [destino / Path(c).name for c in claves]
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda par: self.s3.download_file(self.bucket, par[0], str(par[1])), zip(claves, rutas)))
        return rutas

    def formulario(self, clave: str, max_bytes: int) -> dict:
        return self.s3.generate_presigned_post(self.bucket, clave, Conditions=[["content-length-range", 1, max_bytes]],
                                               ExpiresIn=3600)


ALMACEN = AlmacenS3(WEB_BUCKET, WEB_REGION) if WEB_BUCKET else AlmacenLocal(WEB_DATA_DIR)
TIPOS = {".png": "image/png", ".json": "application/json", ".gz": "application/gzip"}


def escritor(prefijo: str):
    """(escribir, esperar) para ``paquete_web.empaquetar``: sube en paralelo y deja ``caso.json`` al final."""
    pool = ThreadPoolExecutor(max_workers=24)
    pendientes = []

    def subir(nombre: str, datos: bytes) -> None:
        ALMACEN.escribir(f"{prefijo}/{nombre}", datos, TIPOS.get(Path(nombre).suffix))

    def escribir(nombre: str, datos: bytes) -> None:
        if nombre == "caso.json":                       # marca de paquete completo: después de todo lo demás
            for futuro in pendientes:
                futuro.result()
            subir(nombre, datos)
        else:
            pendientes.append(pool.submit(subir, nombre, datos))

    def esperar() -> None:
        for futuro in pendientes:
            futuro.result()
        pool.shutdown()
    return escribir, esperar


def resultado_listo(prefijo: str) -> bool:
    return ALMACEN.existe(f"{prefijo}/caso.json")


# --------------------------------------------------------------------------- trabajos
class Trabajo:
    """Estado de un trabajo en ``trabajos/<id>.json``; la bitácora se guarda como mucho cada 1.5 s."""

    def __init__(self, datos: dict) -> None:
        self.datos = datos
        self.guardado = 0.0

    @classmethod
    def nuevo(cls, tipo: str, **parametros) -> "Trabajo":
        ahora = time.time()
        trabajo = cls({"id": uuid.uuid4().hex, "tipo": tipo, "parametros": parametros, "estado": "en_cola",
                       "log": [], "resultado": None, "error": None, "creado": ahora, "actualizado": ahora})
        trabajo.guardar()
        return trabajo

    @classmethod
    def cargar(cls, id_: str) -> "Trabajo | None":
        if not re.fullmatch(r"[0-9a-f]{32}", id_):
            return None
        datos = ALMACEN.leer(f"trabajos/{id_}.json")
        return cls(json.loads(datos)) if datos else None

    def guardar(self) -> None:
        self.datos["actualizado"] = time.time()
        ALMACEN.escribir(f"trabajos/{self.datos['id']}.json", json.dumps(self.datos).encode(), "application/json")
        self.guardado = time.monotonic()

    def log(self, mensaje: str) -> None:
        print(f"[trabajo {self.datos['id'][:8]}] {mensaje}", flush=True)
        self.datos["log"].append({"t": round(time.time() - self.datos["creado"], 1), "m": str(mensaje)})
        if time.monotonic() - self.guardado > 1.5:
            self.guardar()

    def terminar(self, resultado: str | None = None, error: str | None = None) -> None:
        self.datos.update(estado="error" if error else "listo", resultado=resultado, error=error)
        self.guardar()


def lanzar(trabajo: Trabajo) -> None:
    """En Lambda: invocación asíncrona de esta misma función; en el Mac: un hilo."""
    funcion = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if funcion:
        import boto3
        boto3.client("lambda").invoke(FunctionName=funcion, InvocationType="Event",
                                      Payload=json.dumps({"trabajo": trabajo.datos["id"]}).encode())
    else:
        threading.Thread(target=ejecutar, args=(trabajo.datos["id"],), daemon=True).start()


def ejecutar(id_: str) -> None:
    trabajo = Trabajo.cargar(id_)
    if trabajo is None or trabajo.datos["estado"] != "en_cola":
        return
    trabajo.datos["estado"] = "procesando"
    trabajo.guardar()
    p = trabajo.datos["parametros"]
    try:
        if trabajo.datos["tipo"] == "precalculado":
            resultado = _precalculado(p["caso"], trabajo.log)
        elif trabajo.datos["tipo"] == "vivo":
            resultado = _vivo(p["caso"], p["destino"], p["tta"], trabajo.log)
        else:
            resultado = _subida(p["subida"], p["destino"], p["tta"], p["enderezar"], trabajo.log,
                                p.get("grosor_mm"))
    except Exception as error:                     # se informa en la página tal cual
        traceback.print_exc()
        trabajo.terminar(error=str(error) or error.__class__.__name__)
        return
    trabajo.terminar(resultado=resultado)


def _empaquetar(carpeta: Path, prefijo: str, log) -> str:
    from paquete_web import empaquetar
    log("Preparando el caso para la página (cortes, mallas e IDs)…")
    escribir, esperar = escritor(prefijo)
    empaquetar(carpeta, escribir)
    esperar()
    return prefijo


def _predictor(destino: str, tta: bool):
    """(info, predecir, dónde) como ``choose_predictor`` del dashboard: CPU de esta Lambda o la GPU."""
    if destino == "gpu":
        if not INFERENCE_URL:
            raise RuntimeError("La GPU no está configurada en este servidor (falta INFERENCE_URL).")
        from inferencia_remota import predictor_remoto
        info, predecir = predictor_remoto(INFERENCE_URL, INFERENCE_KEY, destino="gpu")
        return info, predecir, f"AWS · GPU (SageMaker) ({info['precision']})"
    from demo_export import predictor_local
    info, predecir = predictor_local(MODEL_PATH, tta=tta, precision=PRECISION)
    return info, predecir, f"AWS · CPU (Lambda) ({info['device']}, {info['precision']})"


def _slug(destino: str, info: dict, tta: bool) -> str:
    return f"{'aws-gpu' if destino == 'gpu' else 'aws-cpu'}-{info.get('precision', 'fp32')}{'-tta' if info.get('tta') and tta else ''}"


def _precalculado(caso: str, log) -> str:
    import almacen_s3
    prefijo = f"resultados/{MODELO}/pengwin/{caso}"
    if resultado_listo(prefijo):
        return prefijo
    with tempfile.TemporaryDirectory() as tmp:
        log(f"Descargando el caso {caso} precalculado de S3…")
        almacen_s3.descargar_exportado(MODELO, caso, Path(tmp))
        return _empaquetar(Path(tmp), prefijo, log)


def _vivo(caso: str, destino: str, tta: bool, log) -> str:
    import almacen_s3
    from demo_export import exportar_volumen, preparar_caso
    info, predecir, donde = _predictor(destino, tta)
    prefijo = f"resultados/{MODELO}/vivo/{caso}-{_slug(destino, info, tta)}"
    if resultado_listo(prefijo):
        return prefijo
    log(f"Modelo: {info['checkpoint']} · {donde}")
    inicio = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log("Descargando el .mha original de S3…")
        imagen, etiqueta = almacen_s3.descargar_crudo(caso, tmp / "crudo")
        image, label, spacing_xyz, flipped, pasos = preparar_caso(imagen, etiqueta, caso,
                                                                  bool(info.get("canonical")), log=log)
        exportar_volumen(image, spacing_xyz, tmp / "exportado", info, predecir, label=label, nombre=caso,
                         flipped=flipped, pasos=pasos, log=log, max_voxeles_vista=VISTA_MAX_VOXELES,
                         extra_meta={"en_vivo": {"donde": donde}})
        del image, label
        log(f"Pipeline listo en {time.perf_counter() - inicio:.0f} s")
        return _empaquetar(tmp / "exportado", prefijo, log)


def _subida(id_subida: str, destino: str, tta: bool, enderezar: bool, log, grosor_mm: float | None = None) -> str:
    from demo_export import exportar_volumen
    from volumen_entrada import leer_subida
    manifiesto = json.loads(ALMACEN.leer(f"subidas/{id_subida}/manifiesto.json") or b"null")
    if not manifiesto:
        raise RuntimeError("No encontré los archivos subidos (¿expiró la subida?). Vuelve a subir el CT.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log(f"Leyendo {len(manifiesto['archivos'])} archivo(s) subido(s)…")
        ALMACEN.descargar(f"subidas/{id_subida}/archivos", tmp)
        archivos = []
        h = hashlib.sha1()
        for archivo in sorted(manifiesto["archivos"], key=lambda a: a["nombre"]):
            datos = (tmp / archivo["clave"]).read_bytes()
            archivos.append((archivo["nombre"], datos))
            h.update(Path(archivo["nombre"]).name.encode())
            h.update(datos)
        huella = h.hexdigest()[:10]
        info, predecir, donde = _predictor(destino, tta)
        prefijo = (f"resultados/{MODELO}/subidas/{huella}-{_slug(destino, info, tta)}"
                   f"{'' if enderezar else '-sinenderezar'}{f'-g{grosor_mm:g}' if grosor_mm else ''}")
        if resultado_listo(prefijo):
            log("Este CT ya se había procesado con la misma configuración")
            return prefijo
        log(f"Modelo: {info['checkpoint']} · {donde}")
        volume = leer_subida(archivos, orientar=bool(info.get("canonical")), grosor_mm=grosor_mm)
        del archivos
        for ruta in tmp.iterdir():
            ruta.unlink()
        log(f"Leído: {volume.fuente} · {volume.image.shape[0]} cortes de {volume.image.shape[1]}×"
            f"{volume.image.shape[2]} · vóxel {'×'.join(f'{v:.2f}' for v in volume.spacing_xyz)} mm · "
            f"orientación: {volume.orientacion['accion']}")
        inicio = time.perf_counter()
        exportar_volumen(volume.image, volume.spacing_xyz, tmp / "exportado", info, predecir, nombre=volume.nombre,
                         enderezar=enderezar, log=log, pasos=volume.pasos, max_voxeles_vista=VISTA_MAX_VOXELES,
                         extra_meta={"subido": {"fuente": volume.fuente, "orientacion": volume.orientacion,
                                                "donde": donde}})
        del volume
        log(f"Pipeline listo en {time.perf_counter() - inicio:.0f} s")
        return _empaquetar(tmp / "exportado", prefijo, log)


@app.post("/events")
async def eventos(request: Request) -> dict:
    """Invocación asíncrona (no HTTP) que el Lambda Web Adapter entrega aquí: corre el trabajo."""
    datos = await request.json()
    ejecutar(str(datos.get("trabajo", "")))
    return {"ok": True}


# --------------------------------------------------------------------------- casos
_CACHE: dict = {}


def _cacheado(clave: str, segundos: float, calcular):
    valor, hasta = _CACHE.get(clave, (None, 0.0))
    if time.time() >= hasta:
        valor = calcular()
        _CACHE[clave] = (valor, time.time() + segundos)
    return valor


def _gpu_estado() -> dict | None:
    if not INFERENCE_URL:
        return None
    from inferencia_remota import control_gpu
    try:
        datos = control_gpu(INFERENCE_URL, INFERENCE_KEY, "estado")
    except Exception:
        return None
    return datos if datos.get("endpoint") else None


@app.get("/api/config")
def config() -> dict:
    gpu = _cacheado("gpu", 20, _gpu_estado)
    destinos = [{"id": "cpu", "nombre": "AWS · CPU (Lambda)",
                 "detalle": "En este mismo servidor, int8 en CPU. Sin costo extra."}]
    if gpu and gpu.get("estado") == "InService":
        destinos.append({"id": "gpu", "nombre": "AWS · GPU (SageMaker)",
                         "detalle": "NVIDIA T4, fp16. Mucho más rápida para procesar un volumen completo."})
    return {"modelo": MODELO, "destinos": destinos, "gpu": gpu, "max_subida_mb": MAX_SUBIDA_MB,
            "almacen": "s3" if WEB_BUCKET else "local"}


def _casos() -> dict:
    ejemplos = [c for c in ALMACEN.subcarpetas(f"resultados/{MODELO}/ejemplos")]
    try:
        import almacen_s3
        pengwin, error = almacen_s3.listar_casos(MODELO), None
    except Exception as exc:                  # sin credenciales en el Mac: la página lo dice
        pengwin, error = [], f"No pude leer el bucket de PENGWIN: {exc}"
    hechos = set(ALMACEN.subcarpetas(f"resultados/{MODELO}/pengwin"))
    for caso in pengwin:
        caso["listo_web"] = caso["caso"] in hechos
    return {"modelo": MODELO, "ejemplos": [{"caso": c, "prefijo": f"resultados/{MODELO}/ejemplos/{c}"} for c in ejemplos],
            "pengwin": pengwin, "error": error}


@app.get("/api/casos")
def casos() -> dict:
    return _cacheado("casos", 120, _casos)


def _caso_valido(caso: str) -> str:
    if not re.fullmatch(r"\d{3}", caso or ""):
        raise HTTPException(status_code=422, detail="caso inválido")
    return caso


def _destino_valido(destino: str) -> str:
    if destino not in ("cpu", "gpu"):
        raise HTTPException(status_code=422, detail="destino inválido")
    return destino


@app.post("/api/abrir")
async def abrir(request: Request) -> dict:
    """{caso, modo: precalculado|vivo, destino: cpu|gpu, tta} → {resultado} si ya está, o {trabajo}."""
    datos = await request.json()
    caso = _caso_valido(str(datos.get("caso", "")))
    modo = datos.get("modo", "precalculado")
    if modo == "precalculado":
        prefijo = f"resultados/{MODELO}/pengwin/{caso}"
        if resultado_listo(prefijo):
            return {"resultado": prefijo}
        trabajo = Trabajo.nuevo("precalculado", caso=caso)
    elif modo == "vivo":
        trabajo = Trabajo.nuevo("vivo", caso=caso, destino=_destino_valido(datos.get("destino", "cpu")),
                                tta=bool(datos.get("tta")))
    else:
        raise HTTPException(status_code=422, detail="modo inválido")
    lanzar(trabajo)
    return {"trabajo": trabajo.datos["id"]}


@app.get("/api/trabajos/{id_}")
def trabajo_estado(id_: str) -> dict:
    trabajo = Trabajo.cargar(id_)
    if trabajo is None:
        raise HTTPException(status_code=404, detail="no existe ese trabajo")
    datos = dict(trabajo.datos)
    # Una invocación que murió (memoria o 15 min) no alcanza a escribir el error: se detecta por silencio.
    if datos["estado"] in ("en_cola", "procesando") and time.time() - datos["actualizado"] > 16 * 60:
        datos.update(estado="error", error="El procesamiento se interrumpió (tiempo o memoria del servidor).")
    return datos


# --------------------------------------------------------------------------- subir un CT
def _nombre_seguro(nombre: str) -> str:
    base = Path(nombre.replace("\\", "/")).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)[:120] or "archivo"


@app.post("/api/subidas")
async def nueva_subida(request: Request) -> dict:
    """{archivos: [{nombre, tamano}]} → id y un formulario firmado de S3 por archivo."""
    datos = await request.json()
    archivos = datos.get("archivos") or []
    if not archivos or len(archivos) > MAX_ARCHIVOS:
        raise HTTPException(status_code=422, detail=f"hay que subir entre 1 y {MAX_ARCHIVOS} archivos")
    total = sum(int(a.get("tamano", 0)) for a in archivos)
    if total > MAX_SUBIDA_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"la subida pasa de {MAX_SUBIDA_MB} MB")
    id_ = uuid.uuid4().hex
    manifiesto, formularios = [], []
    for indice, archivo in enumerate(archivos):
        clave = f"{indice:05d}_{_nombre_seguro(str(archivo.get('nombre', '')))}"
        manifiesto.append({"nombre": str(archivo.get("nombre", ""))[:300], "clave": clave})
        formularios.append(ALMACEN.formulario(f"subidas/{id_}/archivos/{clave}",
                                              MAX_SUBIDA_MB * 1024 * 1024))
    ALMACEN.escribir(f"subidas/{id_}/manifiesto.json", json.dumps({"archivos": manifiesto}).encode(),
                     "application/json")
    return {"id": id_, "formularios": formularios}


@app.post("/api/subidas/{id_}/procesar")
async def procesar_subida(id_: str, request: Request) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", id_) or not ALMACEN.existe(f"subidas/{id_}/manifiesto.json"):
        raise HTTPException(status_code=404, detail="no existe esa subida")
    datos = await request.json()
    try:
        grosor = float(datos.get("grosor_mm") or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="grosor inválido") from None
    if not 0 <= grosor <= 20:
        raise HTTPException(status_code=422, detail="el grosor entre cortes va de 0 a 20 mm")
    trabajo = Trabajo.nuevo("subida", subida=id_, grosor_mm=grosor or None, destino=_destino_valido(datos.get("destino", "cpu")),
                            tta=bool(datos.get("tta")), enderezar=bool(datos.get("enderezar", True)))
    lanzar(trabajo)
    return {"trabajo": trabajo.datos["id"]}


@app.post("/api/dev/subir")
async def subir_local(request: Request) -> Response:
    """Solo sin S3: recibe el mismo formulario que el POST firmado de S3."""
    if WEB_BUCKET:
        raise HTTPException(status_code=404)
    formulario = await request.form()
    clave = str(formulario["key"])
    if not clave.startswith("subidas/") or ".." in clave:
        raise HTTPException(status_code=422)
    ALMACEN.escribir(clave, await formulario["file"].read())
    return Response(status_code=204)


# --------------------------------------------------------------------------- inferencia en vivo y GPU
@app.post("/api/corte")
async def corte(request: Request, x_destino: str | None = Header(default=None)) -> dict:
    """Un corte uint8 de 384×384 (bytes crudos, fila por fila) → semántica, cajas y latencia."""
    cuerpo = await request.body()
    if len(cuerpo) != TAMANO * TAMANO:
        raise HTTPException(status_code=422, detail=f"se esperan {TAMANO}×{TAMANO} bytes")
    buffer = io.BytesIO()
    np.save(buffer, np.frombuffer(cuerpo, dtype=np.uint8).reshape(TAMANO, TAMANO), allow_pickle=False)
    inicio = time.perf_counter()
    if (x_destino or "cpu") == "gpu":
        if not INFERENCE_URL:
            raise HTTPException(status_code=503, detail="la GPU no está configurada")
        from inferencia_remota import _pedir
        try:
            datos = _pedir(INFERENCE_URL, INFERENCE_KEY, "/predict_slice", buffer.getvalue(), destino="gpu")
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        datos["donde"] = "AWS · GPU (SageMaker)"
    else:
        from api import server
        datos = server._slice(buffer.getvalue())
        datos["donde"] = "AWS · CPU (Lambda)"
    with np.load(io.BytesIO(base64.b64decode(datos.pop("semantic_npz_b64")))) as empaquetado:
        semantica = np.ascontiguousarray(empaquetado["semantic"], dtype=np.uint8)
    datos["semantica_b64"] = base64.b64encode(semantica.tobytes()).decode("ascii")
    datos["servidor_ms"] = (time.perf_counter() - inicio) * 1000.0
    return datos


def _control_gpu(accion: str, clave: str | None = None) -> dict:
    if not INFERENCE_URL:
        raise HTTPException(status_code=404, detail="la GPU no está configurada")
    from inferencia_remota import control_gpu
    try:
        datos = control_gpu(INFERENCE_URL, INFERENCE_KEY, accion, clave)
    except RuntimeError as error:
        codigo = str(error).split(":", 1)[0]
        raise HTTPException(status_code=int(codigo) if codigo.isdigit() else 502,
                            detail=str(error).split(":", 1)[-1].strip()) from error
    _CACHE.pop("gpu", None)
    return datos


@app.get("/api/gpu")
def gpu() -> dict:
    return _control_gpu("estado")


@app.post("/api/gpu/{accion}")
def gpu_accion(accion: str, x_admin_key: str | None = Header(default=None)) -> dict:
    if accion not in ("encender", "apagar"):
        raise HTTPException(status_code=404)
    if not x_admin_key:
        raise HTTPException(status_code=403, detail="falta la clave de administración")
    return _control_gpu(accion, x_admin_key)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "modelo": MODELO}


# En el Mac la API también sirve los resultados y, si ya se compiló (npm run build), la página
# (en AWS los dos los sirve CloudFront desde el bucket).
if not WEB_BUCKET:
    from fastapi.staticfiles import StaticFiles
    for _carpeta in ("resultados", "datos"):
        (WEB_DATA_DIR / _carpeta).mkdir(parents=True, exist_ok=True)
        app.mount(f"/{_carpeta}", StaticFiles(directory=WEB_DATA_DIR / _carpeta), name=_carpeta)
    if (ROOT / "web" / "frontend" / "dist" / "index.html").exists():
        app.mount("/", StaticFiles(directory=ROOT / "web" / "frontend" / "dist", html=True), name="pagina")
