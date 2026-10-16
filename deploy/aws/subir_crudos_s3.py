"""Sube los 100 casos de PENGWIN (imagen y etiqueta) al bucket S3 del dashboard, comprimidos.

    AWS_PROFILE=almia python deploy/aws/subir_crudos_s3.py --bucket pengwin-ct-050752608083 [--casos 001 044]

Cada ``.mha`` se reescribe con compresión (``CompressedData = True``): los vóxeles, la dirección, el
espaciado y la cabecera ``TransformMatrix`` quedan idénticos, y SimpleITK los lee igual. Las imágenes
pasan de ~18.5 GB a ~12 GB y las etiquetas de 13 GB a unos pocos MB. Quedan en
``crudos/imagenes/<caso>.mha`` y ``crudos/etiquetas/<caso>.mha``. Lo que ya está en el bucket con el
mismo tamaño se salta, así que se puede cortar y volver a correr.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pengwin_io import find_image_path, find_label_path  # noqa: E402


def existentes(bucket: str, prefijo: str) -> dict[str, int]:
    salida = subprocess.run(["aws", "s3", "ls", f"s3://{bucket}/{prefijo}"], capture_output=True, text=True).stdout
    return {linea.split()[-1]: int(linea.split()[-2]) for linea in salida.splitlines() if linea.strip()}


def subir(origen: Path, bucket: str, clave: str, ya: dict[str, int], carpeta: Path) -> str:
    destino = carpeta / Path(clave).name
    sitk.WriteImage(sitk.ReadImage(str(origen)), str(destino), useCompression=True)
    tam = destino.stat().st_size
    if ya.get(Path(clave).name) == tam:
        destino.unlink()
        return "ya estaba"
    for intento in range(4):                  # la conexión de la casa a veces se corta a mitad de un archivo
        if subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(destino), f"s3://{bucket}/{clave}"]).returncode == 0:
            break
        if intento == 3:
            raise RuntimeError(f"No se pudo subir {clave} después de 4 intentos")
        time.sleep(15 * (intento + 1))
    destino.unlink()
    return f"{tam / 1e6:.0f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--casos", nargs="*", default=[f"{i:03d}" for i in range(1, 101)])
    args = parser.parse_args()
    ya_img, ya_lbl = existentes(args.bucket, "crudos/imagenes/"), existentes(args.bucket, "crudos/etiquetas/")
    inicio = time.time()
    with tempfile.TemporaryDirectory() as carpeta:
        for n, caso in enumerate(args.casos, 1):
            t = time.time()
            img = subir(find_image_path(caso), args.bucket, f"crudos/imagenes/{caso}.mha", ya_img, Path(carpeta))
            etiqueta = find_label_path(caso)
            lbl = subir(etiqueta, args.bucket, f"crudos/etiquetas/{caso}.mha", ya_lbl, Path(carpeta)) if etiqueta else "sin etiqueta"
            print(f"[{n}/{len(args.casos)}] {caso}: imagen {img}, etiqueta {lbl} ({time.time() - t:.0f} s; "
                  f"total {(time.time() - inicio) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
