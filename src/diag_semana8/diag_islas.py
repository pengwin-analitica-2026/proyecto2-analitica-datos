import sys
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from pengwin_io import load_case, clean_bone_mask, MIN_NOISE_VOXELS

caso = load_case("001", load_label=False)
m = clean_bone_mask(caso.image)
sx, sy, sz = caso.spacing_xyz

lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
objs = ndi.find_objects(lab)
print("MIN_NOISE_VOXELS =", MIN_NOISE_VOXELS, "| componentes:", n)
for k in np.argsort(sizes)[::-1][:8]:
    z, y, x = objs[k]
    print(f"comp {k+1}: {int(sizes[k]):>9,} vox | "
          f"z {z.start*sz:.0f}-{z.stop*sz:.0f} mm | "
          f"y {y.start*sy:.0f}-{y.stop*sy:.0f} mm | "
          f"x {x.start*sx:.0f}-{x.stop*sx:.0f} mm")