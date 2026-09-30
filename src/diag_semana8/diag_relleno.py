import sys
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import ball
from pengwin_io import load_case, body_mask

CASOS = sorted(set([f"{i:03d}" for i in range(1, 101, 5)] + ["025", "068"]))
M = 15
SE = ball(2)

def candidatos(img, cuerpo):
    base = (img > 250) & cuerpo
    h = apply_hysteresis_threshold(img, 150, 250) & cuerpo
    cerrado = ndi.binary_closing(base, structure=SE)
    rel_base = ndi.binary_fill_holes(cerrado) & cuerpo
    cerrado_h = ndi.binary_closing(h, structure=SE)
    rel_h = ndi.binary_fill_holes(cerrado_h) & cuerpo
    return {"250": base, "hist": h, "250+relleno": rel_base, "hist+relleno": rel_h}

print("caso | " + " | ".join(f"{n} rec/prec" for n in ["250", "hist", "250+relleno", "hist+relleno"]))
acc = {n: [] for n in ["250", "hist", "250+relleno", "hist+relleno"]}
for cid in CASOS:
    try:
        caso = load_case(cid, load_label=True)
    except FileNotFoundError:
        continue
    gt = caso.label > 0
    if not gt.any():
        continue
    sl = tuple(slice(max(s.start - M, 0), s.stop + M) for s in ndi.find_objects(gt.astype(np.uint8))[0])
    img, g = caso.image[sl], gt[sl]
    cuerpo = body_mask(caso.image)[sl]
    sx, sy, sz = caso.spacing_xyz
    cerca = ndi.distance_transform_edt(~g, sampling=(sz, sy, sx)) <= 8
    fila = []
    for nombre, m in candidatos(img, cuerpo).items():
        rec = m[g].mean()
        prec = (m & cerca).sum() / max(m.sum(), 1)
        acc[nombre].append((rec, prec))
        fila.append(f"{rec:.2f}/{prec:.2f}")
    print(f"{cid} | " + " | ".join(fila))

print("\nMEDIANA rec/prec")
for n, v in acc.items():
    a = np.array(v)
    print(f"{n}: {np.median(a[:, 0]):.2f} / {np.median(a[:, 1]):.2f}")