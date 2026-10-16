# Experimento PENGWIN: entrena una receta y la valida (15 casos). No toca test.
import base64, csv, glob, io, json, os, pathlib, subprocess, sys, tarfile, time

NOMBRE = "__NOMBRE__"
EXTRA = "__EXTRA__"          # opciones que cambian entre experimentos
SALIDA_PTH = "__PTH__"
PAYLOAD = "__PAYLOAD__"

REPO = pathlib.Path("/kaggle/working/repo")
RES = pathlib.Path("/kaggle/working/resultados")
REPO.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(REPO)
os.chdir(REPO)


def sh(cmd, env=None, check=True):
    print(">>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, env=env)
    if check and r.returncode != 0:
        raise SystemExit(f"FALLO ({r.returncode}): {cmd}")
    return r.returncode


sh("nvidia-smi -L", check=False)
sh("pip -q install SimpleITK scikit-image", check=False)

# Caché: la carpeta que tenga index.csv junto a cache_384
idx = [p for p in glob.glob("/kaggle/input/**/index.csv", recursive=True)
       if os.path.isdir(os.path.join(os.path.dirname(p), "cache_384"))]
assert idx, "No encontré el caché en /kaggle/input"
os.makedirs("data/cache", exist_ok=True)
if not os.path.exists("data/cache/s10"):
    os.symlink(os.path.dirname(idx[0]), "data/cache/s10")
CD = "data/cache/s10/cache_384"
splits = [r["split"] for r in csv.DictReader(open("data/cache/s10/index.csv"))]
conteo = {s: splits.count(s) for s in sorted(set(splits))}
print("Caché:", conteo, "| archivos:", len(glob.glob(CD + "/*.npz")), flush=True)
assert conteo.get("train") == 70 and conteo.get("val") == 15, conteo
assert "test" not in conteo, "este experimento no debe tener test"

BASE = ("--cache-root data/cache/s10 --size 384 --mode stretch --epochs 60 --batch-size 16 --lr 0.002 "
        "--base-channels 32 --cbam 1 --instance-mode primary --augment --cosine --margin 10 --neg-fraction 0.15 "
        "--per-epoch 2500 --val-stride 6 --eval-every 10 --max-minutes 170 --secondary-weight 3.0 "
        "--canonical --aug-geo 1.0 --aug-rot-deg 35")
pth = RES / SALIDA_PTH
t0 = time.time()
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
sh(f'python -u src/train_s10.py {BASE} {EXTRA} --seed 42 --output "{pth}"', env=env0)
print(f"entrenamiento: {(time.time() - t0) / 60:.1f} min", flush=True)
info = json.load(open(str(pth).replace(".pth", ".json")))
print("minutos:", round(info["minutes"]), "| cortado por tiempo:", info["stopped_early"], flush=True)

# Validación: 5 corridas, repartidas en las GPU disponibles
import torch
n_gpu = max(torch.cuda.device_count(), 1)
CORRIDAS = [("base", ""), ("noise10", "--stress noise=10"), ("noise25", "--stress noise=25"),
            ("rotmm60", "--stress rotmm=60"), ("rotmm60_norm", "--stress rotmm=60 --normalizar")]
pendientes = list(CORRIDAS); activos = {}; fallos = []
while pendientes or activos:
    libres = [g for g in range(n_gpu) if g not in activos]
    while pendientes and libres:
        g = libres.pop(0); corrida, extra = pendientes.pop(0)
        out = RES / f"val_{NOMBRE}_{corrida}.json"
        log = open(RES / f"val_{NOMBRE}_{corrida}.log", "w")
        cmd = (f'python src/evaluate_volumes.py --checkpoint "{pth}" --cache-dir {CD} --split val --tta '
               f'{extra} --output "{out}"')
        print(f">> [gpu {g}] {corrida}: {cmd}", flush=True)
        p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
        activos[g] = (p, corrida, out, time.time())
    time.sleep(15)
    for g in list(activos):
        p, corrida, out, ini = activos[g]
        if p.poll() is not None:
            ok = p.returncode == 0 and out.exists()
            print(f"<< {corrida}: {'ok' if ok else 'FALLO'} en {(time.time() - ini) / 60:.1f} min", flush=True)
            if not ok:
                fallos.append(corrida)
                print(open(RES / f"val_{NOMBRE}_{corrida}.log").read()[-3000:], flush=True)
            del activos[g]

M = ["classification_f1_macro", "classification_auc_macro", "detection_iou_mean", "detection_map50",
     "detection_map50_95", "segmentation_dice_macro", "segmentation_iou_macro", "instance_dice_matched",
     "instance_iou_matched", "instance_dice_primary", "instance_dice_secondary", "instance_missed_secondary"]
tabla = {}
for corrida, _ in CORRIDAS:
    f = RES / f"val_{NOMBRE}_{corrida}.json"
    if f.exists():
        r = json.load(open(f))["subsets"]["todo"]
        tabla[corrida] = {k: round(r[k], 4) for k in M if k in r}
        print(corrida, tabla[corrida], flush=True)
json.dump({"nombre": NOMBRE, "extra": EXTRA, "minutos_entrenamiento": info["minutes"],
           "cortado_por_tiempo": info["stopped_early"], "tabla": tabla, "fallos": fallos},
          open(RES / f"resumen_{NOMBRE}.json", "w"), indent=1)
print(f"total: {(time.time() - t0) / 60:.1f} min | fallos: {fallos}", flush=True)
import shutil
os.chdir("/kaggle/working"); shutil.rmtree(REPO, ignore_errors=True)   # la salida del kernel queda solo con resultados/
if fallos:
    raise SystemExit(f"Evaluaciones con fallo: {fallos}")
