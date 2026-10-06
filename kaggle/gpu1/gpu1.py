"""Kernel GPU 1: línea base de w10g_384neg sobre val completa + tiempo por época de entrenamiento."""
import glob, os, subprocess, sys, time, shutil
from pathlib import Path

CODIGO = Path("/kaggle/temp/codigo")
index = glob.glob("/kaggle/input/**/index.csv", recursive=True)
assert index, "no se encontró index.csv en /kaggle/input"
ROOT = Path(index[0]).parent
ckpt = glob.glob("/kaggle/input/**/w10g_384neg.pth", recursive=True)
assert ckpt, "no se encontró el checkpoint"
print("datos:", ROOT, "| checkpoint:", ckpt[0], flush=True)
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
os.chdir(CODIGO)
subprocess.run(["nvidia-smi"], check=False)

def run(cmd):
    print("$", " ".join(cmd), flush=True)
    t = time.time()
    r = subprocess.run(cmd)
    print(f"-> código {r.returncode} en {(time.time() - t) / 60:.1f} min", flush=True)
    return r.returncode

run([sys.executable, "src/evaluate_volumes.py", "--checkpoint", ckpt[0], "--cache-dir", str(ROOT / "cache_384"),
     "--split", "val", "--tta", "--output", str(OUT / "linea_base_val.json")])
run([sys.executable, "src/train_s10.py", "--cache-root", str(ROOT), "--size", "384", "--epochs", "2",
     "--per-epoch", "2000", "--batch-size", "16", "--cbam", "1", "--augment", "--eval-every", "1",
     "--output", str(OUT / "tiempo_epoca.pth")])
for f in OUT.glob("*.pth"):
    f.unlink()   # no hace falta conservar el checkpoint de la prueba de tiempo
