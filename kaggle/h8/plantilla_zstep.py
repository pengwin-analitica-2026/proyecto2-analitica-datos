# Grosor de corte en H8 (solo validación): se conserva 1 de cada k cortes y se comparan los umbrales
# del post-proceso 3D en vóxeles y en mm³, para la distancia de separación y para HD95/ASSD.
import base64, csv, glob, io, json, os, pathlib, shutil, subprocess, tarfile, threading, time

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
assert len({r["case_id"] for r in filas}) == 100

CD = "data/cache/s10/cache_384"
H8 = glob.glob("/kaggle/input/**/h8_aumm_noise01_noema_w32_e60_s42.pth", recursive=True)
assert H8, "No encontré el checkpoint de H8"
H8 = H8[0]


def corrida(script, nombre, extra):
    salida = RES / (nombre + ".json")
    cmd = (f'python src/{script} --checkpoint "{H8}" --cache-dir {CD} --split val --tta {extra} '
           f'--output "{salida}"')
    return (nombre, cmd)


CORRIDAS = []
for k in (1, 2, 3):
    for umbral, opcion in (("vox", ""), ("mm", "--umbral-mm")):
        if k == 1 and umbral == "vox":
            continue                      # ya medido: outputs/h8/validacion/*_val_h8.json
        extra = f"--zstep {k} {opcion}".strip()
        CORRIDAS.append(corrida("infer_volume.py", f"distancias_val_h8_z{k}_{umbral}",
                                extra + " --distance-nofill --surface-metrics"))
        CORRIDAS.append(corrida("metricas_superficie.py", f"superficie_val_h8_z{k}_{umbral}", extra))
COLA_A, COLA_B = CORRIDAS[0::2], CORRIDAS[1::2]
fallos, t0 = [], time.time()


def correr(cola, gpu):
    for nombre, cmd in cola:
        ini = time.time()
        print(f">> [gpu {gpu}] {nombre}", flush=True)
        with open(RES / f"{nombre}.log", "w") as log:
            r = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        ok = r.returncode == 0
        print(f"<< [gpu {gpu}] {nombre}: {'ok' if ok else 'FALLO'} en {(time.time() - ini) / 60:.1f} min", flush=True)
        if not ok:
            fallos.append(nombre)
            print(open(RES / f"{nombre}.log").read()[-3000:], flush=True)
            if nombre.startswith("entrenar"):
                return


import torch
if torch.cuda.device_count() >= 2:
    hilos = [threading.Thread(target=correr, args=(COLA_A, 0)), threading.Thread(target=correr, args=(COLA_B, 1))]
    [h.start() for h in hilos]; [h.join() for h in hilos]
else:
    correr(COLA_B, 0); correr(COLA_A, 0)

M = ["classification_f1_macro", "classification_auc_macro", "detection_iou_mean", "detection_map50",
     "detection_map50_95", "segmentation_dice_macro", "segmentation_iou_macro", "instance_dice_matched",
     "instance_iou_matched"]
for f in sorted(RES.glob("*.json")):
    d = json.load(open(f))
    if "subsets" in d:
        print(f.stem, {k: round(d["subsets"]["todo"][k], 3) for k in M}, flush=True)
    elif isinstance(d.get("resumen"), dict) and "todos" in d["resumen"]:
        print(f.stem, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d["resumen"]["todos"].items()}, flush=True)
print(f"total: {(time.time() - t0) / 60:.1f} min | fallos: {fallos}", flush=True)
os.chdir("/kaggle/working"); shutil.rmtree(REPO, ignore_errors=True)
if fallos:
    raise SystemExit(f"Con fallo: {fallos}")
