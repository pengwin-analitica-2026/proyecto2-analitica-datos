import sys
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from pengwin_io import load_case, clean_bone_mask

CASOS = [f"{i:03d}" for i in range(1, 101, 5)]  # 20 casos

print("caso | vox | z(mm) | llenado | % en hueso etiquetado")
for cid in CASOS:
    try:
        caso = load_case(cid, load_label=True)
    except FileNotFoundError:
        continue
    m = clean_bone_mask(caso.image)
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n == 0:
        continue
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    objs = ndi.find_objects(lab)
    sz = caso.spacing_xyz[2]
    for k in np.argsort(sizes)[::-1][1:]:  # todos menos el más grande
        z, y, x = objs[k]
        caja = (z.stop - z.start) * (y.stop - y.start) * (x.stop - x.start)
        comp = lab == (k + 1)
        pct = 100 * (comp & (caso.label > 0)).sum() / sizes[k]
        print(f"{cid} | {int(sizes[k]):>8,} | {z.start*sz:.0f}-{z.stop*sz:.0f} | "
              f"{sizes[k]/caja:.3f} | {pct:.0f}%")