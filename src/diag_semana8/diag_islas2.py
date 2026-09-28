import sys
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from pengwin_io import load_case, clean_bone_mask

caso = load_case("001", load_label=True)
print("atributos de caso:", list(caso.__dict__.keys()))
label = caso.label  # si el nombre es otro, cámbialo según lo que imprimió arriba

m = clean_bone_mask(caso.image)
lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
for k in range(1, n + 1):
    comp = lab == k
    en_gt = int((comp & (label > 0)).sum())
    print(f"comp {k}: {int(comp.sum()):>9,} vox | dentro del hueso etiquetado: "
          f"{en_gt:,} ({100 * en_gt / comp.sum():.1f}%)")