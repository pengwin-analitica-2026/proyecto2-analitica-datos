"""Genera notebooks/colab/colab_pedidos_profesor.ipynb: los pedidos del profesor en una sola sesión.

Qué corre (en una GPU L4 de Colab, en este orden):

1. Dos entrenamientos a la vez, con la receta exacta de H6 y un solo cambio cada uno:
   ``g1`` (CBAM en los 3 bloques con gamma aprendible: el aporte del CBAM por bloque) y
   ``e1`` (EMA solo en la cabeza de segmentación).
2. Mientras entrenan: HD95 y ASSD de todos los fragmentos de H6 en validación y en test.
3. Al terminar: evaluación de g1 y e1 en validación y cuantización de H6 (en CPU, sin
   entrenamientos corriendo, para que la latencia no salga contaminada).
4. Sube todo a un dataset privado de Kaggle para bajarlo al repo.

El token de Kaggle se lee de los secretos de Colab; nunca se escribe en el notebook.

Uso, desde la raíz del repo:  python colab/hacer_cuaderno_profesor.py
Volver a generarlo después de cambiar src/: el cuaderno lleva una copia del código.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from lanzar import ROOT, current_commit, pack_code

H6 = "h6_auggeo_rot35_canon_ema_w32_e60_s42"
SHA_H6 = hashlib.sha256((ROOT / "outputs" / "s10" / f"{H6}.pth").read_bytes()).hexdigest()
BASE = ("--size 384 --epochs 60 --base-channels 32 --secondary-weight 3 --augment --cosine "
        "--batch-size 16 --per-epoch 2500 --max-minutes 170 --eval-every 10 --seed 42 "
        "--aug-geo 1.0 --aug-rot-deg 35 --canonical")
CORRIDAS = {
    "g1_cbam3g_auggeo_rot35_canon_ema_w32_e60_s42": f"{BASE} --cbam 3 --learnable-gamma --ema 0.999",
    "e1_emaseg_auggeo_rot35_canon_w32_e60_s42": f"{BASE} --cbam 1 --ema 0.999 --ema-scope segmentacion",
}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


cells = [
    md("# PENGWIN: pedidos del profesor (gamma por bloque, EMA en segmentación, HD95/ASSD, cuantización)\n"
       "Ejecutar las celdas en orden en un entorno con GPU L4. Las celdas 5 y 6 corren mientras se "
       "entrena; la 7 y la 8, cuando la celda de espera diga que terminó."),
    code("!nvidia-smi\nimport torch, os; print(torch.__version__, torch.cuda.is_available(), os.cpu_count(), 'CPU')"),
    code(f'''
# 1) Código del repo (embebido; el repo de GitHub es privado)
import base64, io, pathlib, tarfile
CODIGO = pathlib.Path("/content/codigo"); CODIGO.mkdir(exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{pack_code()}"))).extractall(CODIGO)
print("código listo | commit {current_commit()}")
'''),
    code('''
# 2) Credenciales de Kaggle SIN escribirlas en el notebook.
# Colab: ícono de la llave -> secreto KAGGLE_API_TOKEN (activar "acceso desde notebook").
import os, getpass
try:
    from google.colab import userdata
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
except Exception:
    os.environ["KAGGLE_API_TOKEN"] = getpass.getpass("Token de Kaggle: ")
!pip -q install -U kaggle SimpleITK
!kaggle kernels list --mine --page-size 1 | head -3   # si falla aquí, el token no está bien puesto
'''),
    code('''
# 3) Datos: caché 384 (train, val y test) + índice, desde la salida del kernel de datos
!mkdir -p /content/datos
import subprocess, time
for intento in range(4):
    r = subprocess.run('kaggle kernels output gustaveloper/pengwin-s10-datos -p /content/datos --file-pattern "(cache_384/.*)|(index.csv)|(resumen.json)" -q', shell=True)
    if r.returncode == 0: break
    print("reintento", intento + 1); time.sleep(10)
!ls /content/datos | head; du -sh /content/datos
'''),
    code(f'''
# 4) Checkpoint de H6: subir outputs/s10/{H6}.pth desde el repo (botón de archivos o este diálogo)
# El archivo pesa 11 MB: se puede subir entero o en dos partes (h6.pth.part0 y .part1, de `split`).
import hashlib, pathlib, shutil
SALIDA = pathlib.Path("/content/salida"); SALIDA.mkdir(exist_ok=True)
H6 = SALIDA / "{H6}.pth"
if not H6.exists():
    from google.colab import files
    for _ in range(2):
        for nombre in files.upload():
            shutil.move(nombre, SALIDA / nombre)
        if H6.exists():
            break
    partes = sorted(SALIDA.glob("*.part*"))
    if not H6.exists() and partes:
        H6.write_bytes(b"".join(p.read_bytes() for p in partes))
huella = hashlib.sha256(H6.read_bytes()).hexdigest()
assert huella == "{SHA_H6}", "el checkpoint no es el de H6 del repo"
print(H6, H6.stat().st_size // 1024, "KB | sha256 correcto")
'''),
    code(f'''
# 5) Lanzar los dos entrenamientos en segundo plano (comparten la L4, como en la semana 10)
import subprocess
CORRIDAS = {json.dumps(CORRIDAS, indent=4, ensure_ascii=False)}
procesos = {{}}
for nombre, args in CORRIDAS.items():
    log = open(SALIDA / f"{{nombre}}.log", "w")
    cmd = f"python src/train_s10.py --cache-root /content/datos {{args}} --output {{SALIDA}}/{{nombre}}.pth"
    procesos[nombre] = subprocess.Popen(cmd, shell=True, cwd=CODIGO, stdout=log, stderr=subprocess.STDOUT)
    print("lanzado", nombre)
'''),
    code('''
# 6) Mientras entrenan: HD95 y ASSD de TODOS los fragmentos de H6 (mismas opciones que el reporte)
# Test: H6 ya está congelado; esto solo agrega métricas al reporte, no elige nada.
for split in ("val", "test"):
    cmd = (f"cd {CODIGO} && python src/infer_volume.py --checkpoint {H6} --cache-dir /content/datos/cache_384 "
           f"--split {split} --tta --distance-nofill --surface-metrics --output {SALIDA}/superficie_h6_{split}.json")
    !{cmd} 2>&1 | tail -25
'''),
    code('''
# Espera: correr esta celda hasta que diga que los dos terminaron (unos 70-90 min en total)
import time
for nombre, p in procesos.items():
    while p.poll() is None:
        print(nombre, "entrenando...", open(SALIDA / f"{nombre}.log").read().strip().splitlines()[-1][:120])
        time.sleep(300)
    print(nombre, "terminó con código", p.returncode)
!grep -h "γ final" /content/salida/*.log
'''),
    code('''
# 7) Evaluar g1 y e1 en el volumen completo de validación (TTA), igual que H6
for nombre in CORRIDAS:
    cmd = (f"cd {CODIGO} && python src/evaluate_volumes.py --checkpoint {SALIDA}/{nombre}.pth "
           f"--cache-dir /content/datos/cache_384 --split val --tta --output {SALIDA}/{nombre}_val.json")
    !{cmd} 2>&1 | tail -15
import json
g = json.loads((SALIDA / "g1_cbam3g_auggeo_rot35_canon_ema_w32_e60_s42.json").read_text())
print("gamma final por bloque:", g["gamma_final"])
'''),
    code('''
# 8) Cuantización de H6 en CPU (sin entrenamientos corriendo). Se elige con validación.
cmd = (f"cd {CODIGO} && python src/cuantizar.py --checkpoint {H6} --cache-dir /content/datos/cache_384 "
       f"--splits data/splits.json --split val --tta --threads {os.cpu_count()} "
       f"--output {SALIDA}/cuantizacion_h6_val.json")
!{cmd} 2>&1 | grep -v -i warn | tail -12
'''),
    code('''
# 9) Subir resultados a un dataset privado de Kaggle (para bajarlos luego al repo)
import json
(SALIDA / "dataset-metadata.json").write_text(json.dumps({"title": "pengwin-profesor", "id": "gustaveloper/pengwin-profesor", "licenses": [{"name": "CC0-1.0"}]}))
!kaggle datasets version -p /content/salida -m "pedidos del profesor" --dir-mode zip 2>&1 | tail -2 || true
!kaggle datasets create -p /content/salida --dir-mode zip 2>&1 | tail -2
'''),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "colab" / "colab_pedidos_profesor.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(out, out.stat().st_size // 1024, "KB")
