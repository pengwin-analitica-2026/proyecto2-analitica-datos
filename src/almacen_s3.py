"""Los 100 casos de PENGWIN en S3, para elegirlos en el dashboard (sin torch).

Estructura del bucket (``PENGWIN_BUCKET``, por defecto ``pengwin-ct-050752608083`` en us-east-2):

    crudos/imagenes/<caso>.mha          volumen original, comprimido (deploy/aws/subir_crudos_s3.py)
    crudos/etiquetas/<caso>.mha         ground truth, comprimido
    exportados/<modelo>/<caso>/         lo que deja demo_export (meta.json, volume.npz, meshes.npz,
                                        raw_view.npz, pasos.npz), subido por deploy/precalcular_casos.py

El dashboard lista los casos, descarga el exportado si existe («precalculado») o el crudo para
procesarlo en ese momento. Credenciales: las de siempre de boto3 (``AWS_PROFILE`` en local; en
Streamlit Cloud, ``AWS_ACCESS_KEY_ID`` y ``AWS_SECRET_ACCESS_KEY`` de un usuario de solo lectura).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

BUCKET = os.environ.get("PENGWIN_BUCKET", "pengwin-ct-050752608083")
REGION = os.environ.get("PENGWIN_BUCKET_REGION", "us-east-2")
ARCHIVOS_EXPORTADOS = ("meta.json", "volume.npz", "meshes.npz", "raw_view.npz", "pasos.npz")
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def cliente():
    import boto3
    return boto3.client("s3", region_name=REGION)


def _prefijos(prefijo: str) -> list[str]:
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
            cliente().download_file(BUCKET, clave, str(destino / nombre))
        except Exception as error:                     # pasos.npz o raw_view.npz pueden faltar
            if nombre == "meta.json":
                raise FileNotFoundError(f"No está s3://{BUCKET}/{clave}") from error
    return destino


def descargar_crudo(caso: str, destino: Path) -> tuple[Path, Path | None]:
    """Imagen y etiqueta (si existe) del caso. Devuelve sus rutas locales."""
    destino.mkdir(parents=True, exist_ok=True)
    imagen = destino / f"{caso}.mha"
    cliente().download_file(BUCKET, f"crudos/imagenes/{caso}.mha", str(imagen))
    etiqueta = destino / f"{caso}_etiqueta.mha"
    try:
        cliente().download_file(BUCKET, f"crudos/etiquetas/{caso}.mha", str(etiqueta))
    except Exception:
        etiqueta = None
    return imagen, etiqueta


def subir_exportado(carpeta: Path, modelo: str, caso: str) -> None:
    for nombre in ARCHIVOS_EXPORTADOS:
        ruta = carpeta / nombre
        if ruta.exists():
            cliente().upload_file(str(ruta), BUCKET, f"exportados/{modelo}/{caso}/{nombre}")
