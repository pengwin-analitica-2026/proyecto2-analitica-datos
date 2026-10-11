import sys
sys.path.insert(0, "src")
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from pengwin_io import load_case

CASOS = sorted(set([f"{i:03d}" for i in range(1, 101, 5)] + ["025", "068"]))
UMBRALES = [250, 200, 150]
M = 15  # margen en vóxeles alrededor de las etiquetas

print("caso | recall@250 | @200 | @150 | histéresis 150/250 | crece vs 250 | HU mediano etiquetado")
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
    rec = [(img[g] > t).mean() for t in UMBRALES]
    hist = apply_hysteresis_threshold(img, 150, 250)
    crece = 100 * (hist.sum() - (img > 250).sum()) / max((img > 250).sum(), 1)
    print(f"{cid} | {rec[0]:.2f} | {rec[1]:.2f} | {rec[2]:.2f} | "
          f"{hist[g].mean():.2f} | +{crece:.0f}% | {np.median(img[g]):.0f}")