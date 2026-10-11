import sys, time
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from pengwin_io import load_case, clean_bone_mask

CASOS = sorted(set([f"{i:03d}" for i in range(1, 101, 5)] + ["025", "068"]))
F = 2
print("caso | recall_250 | recall_final | agregados | fuga>25mm | seg")
rec_b, rec_f, fugas = [], [], []
for cid in CASOS:
    try:
        caso = load_case(cid, load_label=True)
    except FileNotFoundError:
        continue
    sx, sy, sz = caso.spacing_xyz
    sp = (sz, sy, sx)
    base = clean_bone_mask(caso.image, spacing_zyx=sp, low_threshold=None, fill_cavities=False)
    t = time.time()
    final = clean_bone_mask(caso.image, spacing_zyx=sp)
    seg = time.time() - t
    assert not (base & ~final).any(), "la final no contiene a la base"
    gt = caso.label > 0
    add = final & ~base
    fuga = 0.0
    if add.any():
        d = ndi.distance_transform_edt(~base[::F, ::F, ::F], sampling=tuple(s * F for s in sp))
        zz, yy, xx = np.nonzero(add)
        fuga = 100 * (d[zz // F, yy // F, xx // F] > 25).mean()
    rb, rf = base[gt].mean(), final[gt].mean()
    rec_b.append(rb); rec_f.append(rf); fugas.append(fuga)
    print(f"{cid} | {rb:.2f} | {rf:.2f} | {int(add.sum()):>9,} | {fuga:.1f}% | {seg:.0f}")

print(f"\nMEDIANA recall 250: {np.median(rec_b):.2f} | final: {np.median(rec_f):.2f} | "
      f"fuga máx: {max(fugas):.1f}%")