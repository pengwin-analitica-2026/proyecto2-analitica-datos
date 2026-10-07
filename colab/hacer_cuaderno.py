"""Genera notebooks/colab/colab_s10_v3.ipynb con el código del repo embebido (el repo es privado)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from lanzar import pack_code, current_commit, ROOT

def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(True)}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}

cells = [
 md("# PENGWIN Semana 10 — entrenamiento en Colab (GPU)\nEjecutar celdas en orden. Cambiar solo la celda **Experimento**."),
 code("!nvidia-smi\nimport torch; print(torch.__version__, torch.cuda.is_available())"),
 code(f'''
# 1) Código del repo (embebido; el repo de GitHub es privado)
import base64, io, pathlib, tarfile
CODIGO = pathlib.Path("/content/codigo"); CODIGO.mkdir(exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{pack_code()}"))).extractall(CODIGO)
print("código listo | commit {current_commit()}")
'''),
 code('''
# 2) Credenciales de Kaggle SIN escribirlas en el notebook.
# Colab: ícono de la llave -> agregar secreto KAGGLE_API_TOKEN (activar "acceso desde notebook").
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
# 3) Datos: solo lo que se usa (caché 384 estirado + índice) desde la salida del kernel de datos
!mkdir -p /content/datos
import subprocess, time
for intento in range(4):
    r = subprocess.run('kaggle kernels output gustaveloper/pengwin-s10-datos -p /content/datos --file-pattern "(cache_384/.*)|(index.csv)|(resumen.json)" -q', shell=True)
    if r.returncode == 0: break
    print("reintento", intento + 1); time.sleep(10)
!ls /content/datos | head; du -sh /content/datos
'''),
 code('''
# 4) Experimento (cambiar solo esta celda)
NOMBRE = "colab_cbam1_sec3_full_s42"
ARGS = "--size 384 --epochs 30 --cbam 1 --secondary-weight 3 --augment --cosine --seed 42 --batch-size 16 --max-minutes 150 --eval-every 3"
'''),
 code('''
# 5) Entrenar
import os, subprocess
os.makedirs("/content/salida", exist_ok=True)
cmd = f"cd {CODIGO} && python src/train_s10.py --cache-root /content/datos {ARGS} --output /content/salida/{NOMBRE}.pth"
print(cmd)
!{cmd}
'''),
 code('''
# 6) Evaluar en el volumen completo de validación (TTA)
cmd = f"cd {CODIGO} && python src/evaluate_volumes.py --checkpoint /content/salida/{NOMBRE}.pth --cache-dir /content/datos/cache_384 --split val --tta --output /content/salida/{NOMBRE}_val.json"
!{cmd}
'''),
 code('''
# 7) Subir resultados a un dataset privado de Kaggle para que Claude los lea
import json, pathlib
d = pathlib.Path("/content/salida"); 
(d / "dataset-metadata.json").write_text(json.dumps({"title": "pengwin-s10-colab", "id": "gustaveloper/pengwin-s10-colab", "licenses": [{"name": "CC0-1.0"}]}))
!kaggle datasets version -p /content/salida -m "{NOMBRE}" --dir-mode zip 2>&1 | tail -2 || true
!kaggle datasets create -p /content/salida --dir-mode zip 2>&1 | tail -2
'''),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "colab" / "colab_s10_v3.ipynb"
out.write_text(json.dumps(nb, indent=1)); print(out, out.stat().st_size // 1024, "KB")
