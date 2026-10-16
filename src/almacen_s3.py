"""Los 100 casos de PENGWIN en S3, para elegirlos en el dashboard (sin torch).

Estructura del bucket (``PENGWIN_BUCKET``, por defecto ``pengwin-ct-050752608083`` en us-east-2):

    crudos/imagenes/<caso>.mha          volumen original, comprimido (deploy/aws/subir_crudos_s3.py)
    crudos/etiquetas/<caso>.mha         ground truth, comprimido
    exportados/<modelo>/<caso>/         lo que deja demo_export (meta.json, volume.npz, meshes.npz,
                                        raw_view.npz, pasos.npz), subido por deploy/precalcular_casos.py

El dashboard lista los casos, descarga el exportado si existe («precalculado») o el crudo para
procesarlo en ese momento. Dos formas de llegar al bucket, en este orden:

1. **Directo**, si boto3 encuentra credenciales (``AWS_PROFILE=almia`` en el Mac, o
   ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY`` de un usuario de solo lectura).
2. **Por la API de AWS** (``configurar_api(INFERENCE_URL, INFERENCE_KEY)``), sin credenciales de AWS:
   ``GET /s3/listar`` y ``GET /s3/url`` (enlace firmado de 15 minutos). La Lambda lee el bucket con su
   propio rol. Es lo que usa Streamlit Cloud, que solo tiene la URL y la clave de la API.

Subir (``subir_exportado``) solo se puede en modo directo.
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

BUCKET = os.environ.get("PENGWIN_BUCKET", "pengwin-ct-050752608083")
REGION = os.environ.get("PENGWIN_BUCKET_REGION", "us-east-2")
ARCHIVOS_EXPORTADOS = ("meta.json", "volume.npz", "meshes.npz", "raw_view.npz", "pasos.npz")
ROOT = Path(__file__).resolve().parents[1]


_API: dict = {"url": None, "key": None}


def configurar_api(url: str | None, key: str | None) -> None:
    """Respaldo sin credenciales de AWS: leer el bucket a través de la API (``api/server.py``)."""
    _API.update(url=url, key=key)


@lru_cache(maxsize=1)
def cliente():
    import boto3
    from botocore.config import Config
    # Endpoint regional y firma v4: con el global, los enlaces firmados redirigen a us-east-2 y la
    # firma deja de valer (403).
    return boto3.client("s3", region_name=REGION, endpoint_url=f"https://s3.{REGION}.amazonaws.com",
                        config=Config(signature_version="s3v4"))


@lru_cache(maxsize=1)
def hay_credenciales() -> bool:
    try:
        import boto3
        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def modo() -> str:
    """«directo», «api» o «ninguno»."""
    if hay_credenciales():
        return "directo"
    return "api" if _API["url"] else "ninguno"


def _api(ruta: str, **params) -> dict:
    from inferencia_remota import _pedir
    return _pedir(_API["url"], _API["key"], f"{ruta}?{urllib.parse.urlencode(params)}", timeout=60.0)


def _descargar(clave: str, destino: Path) -> None:
    if modo() == "directo":
        cliente().download_file(BUCKET, clave, str(destino))
        return
    if modo() == "ninguno":
        raise RuntimeError("No hay acceso al bucket: ni credenciales de AWS ni la API (INFERENCE_URL).")
    url = _api("/s3/url", clave=clave)["url"]           # RuntimeError (404) si no existe
    with urllib.request.urlopen(url, timeout=300) as respuesta, open(destino, "wb") as salida:
        shutil.copyfileobj(respuesta, salida, 1 << 20)


def _prefijos(prefijo: str) -> list[str]:
    if modo() == "api":
        return _api("/s3/listar", prefijo=prefijo)["nombres"]
    if modo() == "ninguno":
        raise RuntimeError("No hay acceso al bucket: ni credenciales de AWS ni la API (INFERENCE_URL).")
    return prefijos_directo(prefijo)


def prefijos_directo(prefijo: str) -> list[str]:
    """Lista con boto3, sin pasar por la API (la usa la propia API)."""
    nombres, token = [], None
    while True:
        extra = {"ContinuationToken": token} if token else {}
        r = cliente().list_objects_v2(Bucket=BUCKET, Prefix=prefijo, Delimiter="/", **extra)
        nombres += [p["Prefix"][len(prefijo):].strip("/") for p in r.get("CommonPrefixes", [])]
        nombres += [Path(o["Key"]).name for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return nombres
        token = r["NextContinuationToken"]


def splits() -> dict[str, str]:
    """caso → train/val/test, de ``data/splits.json`` (versionado en el repo)."""
    ruta = ROOT / "data" / "splits.json"
    if not ruta.exists():
        return {}
    data = json.loads(ruta.read_text())
    return {str(c).zfill(3): nombre for nombre in ("train", "val", "test") for c in data.get(nombre, [])}


def listar_casos(modelo: str) -> list[dict]:
    """Los casos del bucket: si hay crudo, si está precalculado para ``modelo``, split y si viene girado."""
    from orientacion import is_rotated
    crudos = {Path(n).stem for n in _prefijos("crudos/imagenes/") if n.endswith(".mha")}
    exportados = set(_prefijos(f"exportados/{modelo}/"))
    split = splits()
    return [{"caso": c, "split": split.get(c, "?"), "girado": is_rotated(c), "crudo": c in crudos,
             "precalculado": c in exportados} for c in sorted(crudos | exportados)]


def descargar_exportado(modelo: str, caso: str, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in ARCHIVOS_EXPORTADOS:
        clave = f"exportados/{modelo}/{caso}/{nombre}"
        try:
            _descargar(clave, destino / nombre)
        except Exception as error:                     # pasos.npz o raw_view.npz pueden faltar
            if nombre == "meta.json":
                raise FileNotFoundError(f"No está s3://{BUCKET}/{clave}") from error
    return destino


def descargar_crudo(caso: str, destino: Path) -> tuple[Path, Path | None]:
    """Imagen y etiqueta (si existe) del caso. Devuelve sus rutas locales."""
    destino.mkdir(parents=True, exist_ok=True)
    imagen = destino / f"{caso}.mha"
    _descargar(f"crudos/imagenes/{caso}.mha", imagen)
    etiqueta = destino / f"{caso}_etiqueta.mha"
    try:
        _descargar(f"crudos/etiquetas/{caso}.mha", etiqueta)
    except Exception:
        etiqueta = None
    return imagen, etiqueta


def subir_exportado(carpeta: Path, modelo: str, caso: str) -> None:
    for nombre in ARCHIVOS_EXPORTADOS:
        ruta = carpeta / nombre
        if ruta.exists():
            cliente().upload_file(str(ruta), BUCKET, f"exportados/{modelo}/{caso}/{nombre}")
