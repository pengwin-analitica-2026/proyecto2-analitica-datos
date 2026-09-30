import sys
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from pengwin_io import load_case, clean_bone_mask, body_mask

GRANDE = 50_000
F = 2  # submuestreo para ahorrar memoria
CASOS = [f"{i:03d}" for i in range(1, 101, 5)]
dist_gt = []

print("== A) profundidad de componentes pequeños dentro del cuerpo ==")
print("caso | vox | prof_min(mm) | prof_mediana(mm) | % en hueso etiquetado")
for cid in CASOS:
    try:
        caso = load_case(cid, load_label=True)
    except FileNotFoundError:
        continue
    sx, sy, sz = caso.spacing_xyz
    samp = (sz * F, sy * F, sx * F)

    m = clean_bone_mask(caso.image)
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    lab_ds = lab[::F, ::F, ::F]
    gt_ds = caso.label[::F, ::F, ::F]

    body = body_mask(caso.image)[::F, ::F, ::F]
    prof = ndi.distance_transform_edt(body, sampling=samp)
    for k in np.flatnonzero(sizes < GRANDE) + 1:
        comp = lab_ds == k
        if not comp.any():
            continue
        d = prof[comp]
        print(f"{cid} | {int(sizes[k-1]):>8,} | {d.min():>6.1f} | "
              f"{np.median(d):>6.1f} | {100 * (gt_ds[comp] > 0).mean():.0f}%")

    # B) separación real de cada fragmento respecto al principal de su hueso
    for r in range(3):
        princ = gt_ds == 10 * r + 1
        if not princ.any():
            continue
        d_to_p = ndi.distance_transform_edt(~princ, sampling=samp)
        for f in range(10 * r + 2, 10 * r + 11):
            fm = gt_ds == f
            if fm.any():
                dist_gt.append(d_to_p[fm].min())

dg = np.array(dist_gt)
print("\n== B) distancia real de cada fragmento al principal de su hueso ==")
if len(dg):
    print(f"fragmentos: {len(dg)} | mediana {np.median(dg):.1f} mm | "
          f"p75 {np.percentile(dg, 75):.1f} | p90 {np.percentile(dg, 90):.1f} | "
          f"máx {dg.max():.1f}")
    print(f"a más de 3.5 mm del principal: {100 * (dg > 3.5).mean():.0f}%")
    