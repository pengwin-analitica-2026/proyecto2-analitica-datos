"""Kernel GPU: receta exacta de H6 con CBAM en los 3 bloques y gamma aprendible por bloque.

Experimento pedido por el profesor: medir el aporte del CBAM en cada bloque con el gamma final
(``salida = x + gamma * (CBAM(x) - x)``, gamma inicial 0). No reemplaza a H6; se compara en
validación. El gamma final queda en <NOMBRE>.json (claves ``gamma_final`` y ``gamma_checkpoint``).

Cambiar NOMBRE y ARGS (o generarlo con sed antes de lanzar). Deja en /kaggle/working:
<NOMBRE>.pth, <NOMBRE>.json (historial de validación) y <NOMBRE>_val.json (volumen completo, TTA).
"""
import glob, os, subprocess, sys, time
from pathlib import Path

NOMBRE = "g1_cbam3g_auggeo_rot35_canon_ema_w32_e60_s42"
ARGS = "--size 384 --epochs 60 --base-channels 32 --cbam 3 --learnable-gamma --secondary-weight 3 --augment --cosine --batch-size 16 --per-epoch 2500 --max-minutes 170 --eval-every 10 --ema 0.999 --seed 42 --aug-geo 1.0 --aug-rot-deg 35 --canonical"
EVALUAR = True

CODIGO = Path("/kaggle/temp/codigo")
index = glob.glob("/kaggle/input/**/index.csv", recursive=True)
assert index, "no se encontró index.csv en /kaggle/input"
ROOT = Path(index[0]).parent
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
os.chdir(CODIGO)
subprocess.run(["nvidia-smi"], check=False)

def run(cmd):
    print("$", " ".join(cmd), flush=True)
    t = time.time(); r = subprocess.run(cmd)
    print(f"-> código {r.returncode} en {(time.time() - t) / 60:.1f} min", flush=True)
    return r.returncode

ok = run([sys.executable, "src/train_s10.py", "--cache-root", str(ROOT), *ARGS.split(),
          "--output", str(OUT / f"{NOMBRE}.pth")])
if ok == 0 and EVALUAR:
    size = ARGS.split()[ARGS.split().index("--size") + 1] if "--size" in ARGS else "384"
    mode = ARGS.split()[ARGS.split().index("--mode") + 1] if "--mode" in ARGS else "stretch"
    carpeta = f"cache_{size}" + ("_pad" if mode == "pad" else "")
    run([sys.executable, "src/evaluate_volumes.py", "--checkpoint", str(OUT / f"{NOMBRE}.pth"),
         "--cache-dir", str(ROOT / carpeta), "--split", "val", "--tta",
         "--output", str(OUT / f"{NOMBRE}_val.json")])
