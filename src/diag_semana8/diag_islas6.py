import sys
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from pengwin_io import load_case, clean_bone_mask

CASOS = [f"{i:03d}" for i in range(2, 101, 5)]
for cid in CASOS:
    try:
        caso = load_case(cid, load_label=True)
    except FileNotFoundError:
        continue
    sx, sy, sz = caso.spacing_xyz
    antes = clean_bone_mask(caso.image)
    despues = clean_bone_mask(caso.image, spacing_zyx=(sz, sy, sx))
    lab, n = ndi.label(antes & ~despues, structure=np.ones((3, 3, 3)))
    if n == 0:
        print(cid, "| nada removido")
        continue
    for k in range(1, n + 1):
        c = lab == k
        print(f"{cid} | quitado {int(c.sum()):>8,} vox | "
              f"{100 * (caso.label[c] > 0).mean():.0f}% en hueso etiquetado")