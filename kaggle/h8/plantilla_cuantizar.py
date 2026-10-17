# Cuantización de H8: unas variantes (fp32, int8, int8_todo, bf16, fp16) sobre los 15 casos de un split
# con TTA. int8 de PyTorch solo corre en CPU (y la demo corre en CPU en Lambda); fp16 y bf16 se miden en
# GPU, que es donde rinden.
import base64, glob, io, os, pathlib, platform, subprocess, sys, tarfile

VARIANTES = "__VARIANTES__"     # separadas por espacio
DISPOSITIVO = "__DISPOSITIVO__"  # cpu o cuda
SPLIT = "__SPLIT__"              # val o test
PAYLOAD = "__PAYLOAD__"

REPO = pathlib.Path("/kaggle/working/repo")
REPO.mkdir(parents=True, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(REPO)
os.chdir(REPO)


def sh(cmd, check=True):
    print(">>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FALLO ({r.returncode}): {cmd}")


sh("lscpu | head -20", check=False)
sh("nvidia-smi -L", check=False)
sh("pip -q install SimpleITK scikit-image", check=False)

# Dataset plano (pengwin-cuantizacion-h8 o -test): <caso>.npz, index.csv, splits.json y el .pth.
indices = glob.glob("/kaggle/input/**/index.csv", recursive=True)
assert indices, "No encontré index.csv en /kaggle/input"
DATOS = pathlib.Path(indices[0]).parent
cache = pathlib.Path("data/cache/s10/cache_384")
cache.mkdir(parents=True, exist_ok=True)
for npz in DATOS.glob("*.npz"):
    (cache / npz.name).symlink_to(npz)
pathlib.Path("data/cache/s10/index.csv").symlink_to(DATOS / "index.csv")
pathlib.Path("data/splits.json").symlink_to(DATOS / "splits.json")
pth = next(DATOS.glob("h8_*.pth"))
print("casos en el caché:", len(list(cache.glob("*.npz"))), "| checkpoint:", pth.name, flush=True)

import torch
salida = f"/kaggle/working/{SPLIT}_h8_{VARIANTES.replace(' ', '-')}_{DISPOSITIVO}.json"
sh(f"{sys.executable} -u src/cuantizar.py --checkpoint {pth} --cache-dir {cache} --split {SPLIT} --tta "
   f"--threads {os.cpu_count()} --variants {VARIANTES} --device {DISPOSITIVO} --output {salida}")
print("torch", torch.__version__, "| CPU", platform.processor(), os.cpu_count(), "núcleos | listo:", salida, flush=True)
