# H8 con otra semilla: misma receta, entrenamiento y evaluación (validación con estrés y test).
# Sirve para medir la variación entre semillas; el modelo entregado sigue siendo la semilla 42.
import base64, csv, glob, io, json, os, pathlib, shutil, subprocess, tarfile, time

SEMILLA = int("__SEMILLA__")
PAYLOAD = "__PAYLOAD__"
REPO = pathlib.Path("/kaggle/working/repo")
RES = pathlib.Path("/kaggle/working/resultados")
REPO.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(REPO)
os.chdir(REPO)
subprocess.run("nvidia-smi -L", shell=True)
subprocess.run("pip -q install SimpleITK scikit-image", shell=True)

# Caché: se unen las carpetas de entrenamiento+validación y de test en una sola
indices = [p for p in glob.glob("/kaggle/input/**/index.csv", recursive=True)
           if os.path.isdir(os.path.join(os.path.dirname(p), "cache_384"))]
assert len(indices) == 2, indices
os.makedirs("data/cache/s10/cache_384", exist_ok=True)
filas, campos = [], None
for idx in indices:
    lector = csv.DictReader(open(idx)); campos = lector.fieldnames
    filas += list(lector)
    for npz in glob.glob(os.path.join(os.path.dirname(idx), "cache_384", "*.npz")):
        destino = os.path.join("data/cache/s10/cache_384", os.path.basename(npz))
        if not os.path.exists(destino):
            os.symlink(npz, destino)
filas.sort(key=lambda r: r["case_id"])
with open("data/cache/s10/index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(filas)
conteo = {s: sum(r["split"] == s for r in filas) for s in ("train", "val", "test")}
print("Caché:", conteo, "| archivos:", len(glob.glob("data/cache/s10/cache_384/*.npz")), flush=True)
assert conteo == {"train": 70, "val": 15, "test": 15}, conteo

CD = "data/cache/s10/cache_384"
NOMBRE = f"h8_s{SEMILLA}"
PTH = RES / f"h8_aumm_noise01_noema_w32_e60_s{SEMILLA}.pth"
# Receta exacta de H8 (outputs/h8/h8_aumm_noise01_noema_w32_e60_s42.json); solo cambia la semilla.
ENTRENAR = ("python -u src/train_s10.py --cache-root data/cache/s10 --size 384 --mode stretch --epochs 60 "
            "--batch-size 16 --lr 0.002 --base-channels 32 --cbam 1 --instance-mode primary --augment --cosine "
            "--margin 10 --neg-fraction 0.15 --per-epoch 2500 --val-stride 6 --eval-every 10 --max-minutes 170 "
            "--secondary-weight 3.0 --canonical --aug-geo 1.0 --aug-rot-deg 35 --aug-mm --ema 0 --aug-noise 0.1 "
            f'--seed {SEMILLA} --output "{PTH}"')

t0 = time.time()
print(">> entrenamiento", flush=True)
r = subprocess.run(ENTRENAR, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
if r.returncode != 0:
    raise SystemExit(f"FALLO el entrenamiento ({r.returncode})")
info = json.load(open(str(PTH).replace(".pth", ".json")))
print(f"entrenamiento: {info['minutes']:.1f} min | cortado por tiempo: {info['stopped_early']}", flush=True)

CORRIDAS = [("val_base", "val", ""), ("val_noise10", "val", "--stress noise=10"),
            ("val_noise25", "val", "--stress noise=25"), ("val_rotmm60", "val", "--stress rotmm=60"),
            ("val_rotmm60_norm", "val", "--stress rotmm=60 --normalizar"), ("test", "test", "")]
import torch
n_gpu = max(torch.cuda.device_count(), 1)
pendientes, activos, fallos = list(CORRIDAS), {}, []
while pendientes or activos:
    libres = [g for g in range(n_gpu) if g not in activos]
    while pendientes and libres:
        g = libres.pop(0); corrida, split, extra = pendientes.pop(0)
        out = RES / f"{corrida}_{NOMBRE}.json"
        cmd = (f'python src/evaluate_volumes.py --checkpoint "{PTH}" --cache-dir {CD} --split {split} --tta '
               f'{extra} --output "{out}"')
        print(f">> [gpu {g}] {corrida}", flush=True)
        log = open(RES / f"{corrida}_{NOMBRE}.log", "w")
        p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
        activos[g] = (p, corrida, out, time.time())
    time.sleep(15)
    for g in list(activos):
        p, corrida, out, ini = activos[g]
        if p.poll() is not None:
            ok = p.returncode == 0 and out.exists()
            print(f"<< [gpu {g}] {corrida}: {'ok' if ok else 'FALLO'} en {(time.time() - ini) / 60:.1f} min", flush=True)
            if not ok:
                fallos.append(corrida)
                print(open(RES / f"{corrida}_{NOMBRE}.log").read()[-3000:], flush=True)
            del activos[g]

M = ["classification_f1_macro", "classification_auc_macro", "detection_iou_mean", "detection_map50",
     "detection_map50_95", "segmentation_dice_macro", "segmentation_iou_macro", "instance_dice_matched",
     "instance_iou_matched", "instance_dice_primary", "instance_dice_secondary", "instance_missed_secondary"]
tabla = {}
for corrida, _, _ in CORRIDAS:
    f = RES / f"{corrida}_{NOMBRE}.json"
    if f.exists():
        tabla[corrida] = {k: round(json.load(open(f))["subsets"]["todo"][k], 4) for k in M}
        print(corrida, tabla[corrida], flush=True)
json.dump({"semilla": SEMILLA, "minutos_entrenamiento": info["minutes"],
           "cortado_por_tiempo": info["stopped_early"], "tabla": tabla, "fallos": fallos},
          open(RES / f"resumen_{NOMBRE}.json", "w"), indent=1)
print(f"total: {(time.time() - t0) / 60:.1f} min | fallos: {fallos}", flush=True)
os.chdir("/kaggle/working"); shutil.rmtree(REPO, ignore_errors=True)
if fallos:
    raise SystemExit(f"Con fallo: {fallos}")
