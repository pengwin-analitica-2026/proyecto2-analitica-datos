"""Kernel de Kaggle (CPU, con internet): baja PENGWIN de Zenodo y construye los cachés por caso.

No usa GPU. Deja en /kaggle/working:

    cache_256/  cache_384/           modo stretch (compatible con las semanas 9 y 10)
    cache_256_pad/  cache_384_pad/   modo pad (conserva la proporción)
    labels_native/                   etiquetas en resolución original
    index.csv  resumen.json

Los volúmenes crudos se extraen caso por caso en /kaggle/temp y se borran al
terminar cada caso, porque descomprimidos pesan más de 30 GB.
"""

import glob
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ZENODO = "https://zenodo.org/records/10927452/files/{name}?download=1"
ZIPS = {
    "PENGWIN_CT_train_images_part1.zip": 3944991549,
    "PENGWIN_CT_train_images_part2.zip": 3635398992,
    "PENGWIN_CT_train_labels.zip": 33326051,
}
SIZES = (256, 384)
MODES = ("stretch", "pad")
WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
CODE = TEMP / "codigo"
ZIP_DIR = TEMP / "zips"
RAW = TEMP / "raw"


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def find_code() -> None:
    """Deja el código del repo en /kaggle/temp/codigo, venga como tar o como carpeta."""
    if (CODE / "src" / "build_case_cache.py").exists():
        log("código ya extraído por kaggle/lanzar.py")
        return
    CODE.mkdir(parents=True, exist_ok=True)
    tars = glob.glob("/kaggle/input/**/codigo.tar.gz", recursive=True)
    if tars:
        with tarfile.open(tars[0]) as tar:
            tar.extractall(CODE)
        log("código desde", tars[0])
    else:
        hits = glob.glob("/kaggle/input/**/src/pengwin_io.py", recursive=True)
        assert hits, "No encuentro el código: lanza este script con kaggle/lanzar.py"
        root = Path(hits[0]).parents[1]
        shutil.copytree(root, CODE, dirs_exist_ok=True)
        log("código desde", root)
    assert (CODE / "src" / "build_case_cache.py").exists()


def download(name: str, expected: int) -> Path:
    """Descarga con reanudación y reintentos; verifica el tamaño exacto."""
    target = ZIP_DIR / name
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 9):
        have = target.stat().st_size if target.exists() else 0
        if have == expected:
            return target
        if have > expected:
            target.unlink(); have = 0
        try:
            request = urllib.request.Request(ZENODO.format(name=name), headers={"Range": f"bytes={have}-"} if have else {})
            started, last = time.time(), time.time()
            with urllib.request.urlopen(request, timeout=120) as response, open(target, "ab" if have else "wb") as out:
                if have and response.status != 206:
                    out.close(); target.unlink(); raise RuntimeError("el servidor no reanudó; se reinicia")
                while True:
                    chunk = response.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
                    if time.time() - last > 60:
                        done = target.stat().st_size
                        log(f"  {name}: {done / 1e9:.2f}/{expected / 1e9:.2f} GB, {(done - have) / 1e6 / (time.time() - started):.1f} MB/s")
                        last = time.time()
        except Exception as error:  # la red de Zenodo corta a veces: se reanuda
            log(f"  intento {attempt} de {name}: {error!r}")
            time.sleep(min(60, 5 * attempt))
    size = target.stat().st_size if target.exists() else 0
    assert size == expected, f"{name}: {size} bytes, se esperaban {expected}"
    return target


def members_by_case(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        return {Path(info.filename).stem: info.filename for info in archive.infolist()
                if info.filename.endswith(".mha")}


def process_case(job):
    case_id, split, image_zip, image_member, image_folder, label_zip, label_member = job
    sys.path.insert(0, str(CODE / "src"))
    import build_case_cache as bcc
    case_raw = RAW / case_id
    targets = [(image_zip, image_member, case_raw / image_folder / f"{case_id}.mha"),
               (label_zip, label_member, case_raw / "PENGWIN_CT_train_labels" / f"{case_id}.mha")]
    try:
        for zip_path, member, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive, archive.open(member) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out, 1 << 22)
        return bcc.build_case(case_id, str(WORK), SIZES, bcc.GAMMA, split,
                              raw_root=str(case_raw), modes=MODES)
    finally:
        shutil.rmtree(case_raw, ignore_errors=True)


def main():
    started = time.time()
    log("disco:", {p: f"{shutil.disk_usage(p).free / 1e9:.0f} GB libres" for p in ("/kaggle/working", "/kaggle/temp" if TEMP.exists() else "/tmp")})
    TEMP.mkdir(parents=True, exist_ok=True)
    find_code()
    sys.path.insert(0, str(CODE / "src"))
    import build_case_cache as bcc

    splits = json.loads((CODE / "data" / "splits.json").read_text())
    paths = {}
    for name, expected in ZIPS.items():
        t0 = time.time()
        paths[name] = download(name, expected)
        log(f"descargado {name} en {time.time() - t0:.0f} s")
    part1 = members_by_case(paths["PENGWIN_CT_train_images_part1.zip"])
    part2 = members_by_case(paths["PENGWIN_CT_train_images_part2.zip"])
    labels = members_by_case(paths["PENGWIN_CT_train_labels.zip"])
    log(f"casos en zips: part1={len(part1)} part2={len(part2)} labels={len(labels)}")

    jobs = []
    for split in ("train", "val", "test"):
        for number in splits[split]:
            case_id = str(number).zfill(3)
            if case_id in part1:
                image = (paths["PENGWIN_CT_train_images_part1.zip"], part1[case_id], "PENGWIN_CT_train_images_part1")
            else:
                image = (paths["PENGWIN_CT_train_images_part2.zip"], part2[case_id], "PENGWIN_CT_train_images_part2")
            jobs.append((case_id, split, *image, paths["PENGWIN_CT_train_labels.zip"], labels[case_id]))
    log(f"casos a procesar: {len(jobs)}")

    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(process_case, jobs):
            rows.append(row)
            if len(rows) % 10 == 0 or len(rows) == len(jobs):
                log(f"{len(rows)}/{len(jobs)} casos | último {row['case_id']} ({row['n_slices']} cortes)")
    index = bcc.write_index(rows, WORK)

    summary = {"casos": len(rows), "sizes": SIZES, "modes": MODES, "gamma": bcc.GAMMA,
               "minutos": round((time.time() - started) / 60, 1), "por_split": {}, "carpetas": {}}
    for split in ("train", "val", "test"):
        chosen = [r for r in rows if r["split"] == split]
        summary["por_split"][split] = {"casos": len(chosen),
                                       "cortes": sum(r["n_slices"] for r in chosen),
                                       "cortes_con_hueso": sum(r["n_bone_slices"] for r in chosen)}
    for folder in sorted(p for p in WORK.iterdir() if p.is_dir()):
        files = list(folder.glob("*.npz"))
        summary["carpetas"][folder.name] = {"archivos": len(files),
                                            "GB": round(sum(f.stat().st_size for f in files) / 1e9, 2)}
        assert len(files) == len(rows), f"{folder.name}: {len(files)} archivos para {len(rows)} casos"
    (WORK / "resumen.json").write_text(json.dumps(summary, indent=2))
    log("RESUMEN", json.dumps(summary, indent=2))
    log("índice:", index)
    shutil.rmtree(TEMP, ignore_errors=True)


if __name__ == "__main__":
    main()
