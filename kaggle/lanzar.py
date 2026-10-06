"""Lanza un script de este repo como kernel privado de Kaggle, con el código embebido.

El repo de GitHub es privado, así que en vez de clonarlo se empaqueta ``src/``,
``tests/`` y los archivos pequeños de ``data/`` dentro del propio script que se
envía. En Kaggle quedan extraídos en ``/kaggle/temp/codigo``.

Uso (desde la raíz del repo, con el token en ~/.kaggle/access_token):

    python kaggle/lanzar.py kaggle/datos/construir_datos.py --slug pengwin-s10-datos
    python kaggle/lanzar.py kaggle/x.py --slug pengwin-s10-x --gpu --fuentes gustaveloper/pengwin-s10-datos

Luego:

    kaggle kernels status gustaveloper/<slug>
    kaggle kernels output gustaveloper/<slug> -p outputs/s10/<slug>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "tests", "data/splits.json", "data/manifest.csv", "requirements.txt"]

BOOTSTRAP = '''# --- generado por kaggle/lanzar.py: extrae el código del repo ---
import base64 as _b64, io as _io, pathlib as _pl, tarfile as _tar
_codigo = _pl.Path("/kaggle/temp/codigo"); _codigo.mkdir(parents=True, exist_ok=True)
_tar.open(fileobj=_io.BytesIO(_b64.b64decode("{payload}"))).extractall(_codigo)
print("código extraído en", _codigo, "| commit:", "{commit}")
# --- fin del arranque ---
'''


def pack_code() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in INCLUDE:
            path = ROOT / item
            if path.exists():
                tar.add(path, arcname=item,
                        filter=lambda info: None if "__pycache__" in info.name else info)
    return base64.b64encode(buffer.getvalue()).decode()


def current_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "sin-git"


def build_bundle(script: Path) -> str:
    return BOOTSTRAP.format(payload=pack_code(), commit=current_commit()) + "\n" + script.read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("script")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--usuario", default="gustaveloper")
    parser.add_argument("--gpu", action="store_true", help="Corre con T4 x2; gasta cupo semanal.")
    parser.add_argument("--sin-internet", action="store_true")
    parser.add_argument("--fuentes", nargs="*", default=[], help="Kernels cuya salida se monta en /kaggle/input.")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--timeout", type=int, default=None, help="Segundos máximos de ejecución.")
    parser.add_argument("--solo-armar", action="store_true", help="Escribe el paquete y no lo envía.")
    args = parser.parse_args()

    folder = Path(tempfile.mkdtemp(prefix="kaggle_"))
    (folder / "script.py").write_text(build_bundle(Path(args.script)))
    metadata = {
        "id": f"{args.usuario}/{args.slug}", "title": args.slug, "code_file": "script.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true" if args.gpu else "false",
        "enable_internet": "false" if args.sin_internet else "true",
        "dataset_sources": args.datasets, "competition_sources": [], "kernel_sources": args.fuentes,
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    size_kb = (folder / "script.py").stat().st_size / 1024
    print(f"paquete: {folder} | script {size_kb:.0f} KB | gpu={args.gpu}")
    if args.solo_armar:
        return
    command = ["kaggle", "kernels", "push", "-p", str(folder)]
    if args.timeout:
        command += ["-t", str(args.timeout)]
    if args.gpu:
        command += ["--accelerator", "NvidiaTeslaT4"]
    print(subprocess.run(command, capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
