import sys
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from pengwin_io import load_case, clean_bone_mask

GRANDE = 50_000
CASOS = [f"{i:03d}" for i in range(1, 101, 5)]

print("caso | vox | dist_al_esqueleto(mm) | HU_medio | % en hueso etiquetado")
for cid in CASOS:
    try:
        caso = load_case(cid, load_label=True)
    except FileNotFoundError:
        continue
    m = clean_bone_mask(caso.image)
    if m.sum() == 0:
        continue
    sl = ndi.find_objects(m.astype(np.uint8))[0]   # recorte para ahorrar memoria
    m, img, gt = m[sl], caso.image[sl], caso.label[sl]
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    grandes = np.flatnonzero(sizes >= GRANDE) + 1
    pequenos = np.flatnonzero(sizes < GRANDE) + 1
    if len(grandes) == 0 or len(pequenos) == 0:
        continue
    sx, sy, sz = caso.spacing_xyz
    dist = ndi.distance_transform_edt(~np.isin(lab, grandes), sampling=(sz, sy, sx))
    for k in pequenos:
        comp = lab == k
        print(f"{cid} | {int(sizes[k-1]):>8,} | {dist[comp].min():>7.1f} | "
              f"{img[comp].mean():>6.0f} | {100 * (gt[comp] > 0).mean():.0f}%")