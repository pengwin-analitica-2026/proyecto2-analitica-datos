"""Valida src/normalizacion.py con los volúmenes reales (necesita data/raw).

Hace dos cosas, sin modelo ni etiquetas:

1. **Diagnóstico por caso:** qué dice la cabecera (productos punto y cruz) y qué giro axial estima
   el paso 2 en cada volumen, ya en orientación estándar.
2. **Giros conocidos:** a la silueta de cada caso le aplica giros de ±10, ±20, ±30 y ±60 grados y
   mide si el paso 2 los recupera (error = estimado − real, en grados). Es la misma idea de
   validación del artículo de Ferro Sánchez et al.: medir el ángulo que queda tras corregir.

Uso (desde la raíz del repo):

    python src/validar_normalizacion.py --split val
    python src/validar_normalizacion.py --cases 012 039 044 --output outputs/normalizacion/val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import normalizacion as nz  # noqa: E402
import pengwin_io  # noqa: E402

GIROS = (-60, -30, -20, -10, 10, 20, 30, 60)


def casos_de(split: str) -> list[str]:
    splits = json.loads(Path("data/splits.json").read_text())
    return [str(c).zfill(3) for c in splits[split]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--output", default="outputs/normalizacion/validacion.json")
    args = parser.parse_args()

    casos = args.cases or casos_de(args.split)
    filas, errores = [], {g: [] for g in GIROS}
    for caso in casos:
        img = sitk.ReadImage(str(pengwin_io.find_image_path(caso)))
        estandar, _, diag = nz.a_orientacion_estandar(img)
        vol = sitk.GetArrayFromImage(estandar)
        silueta = nz.mip_axial_hueso(vol, estandar.GetSpacing())
        base = nz.estimar_giro_axial(silueta)
        fila = {"caso": caso, "espejado": diag.espejado, "oblicuo": diag.oblicuo,
                "signos": diag.signos, "giro_estimado": round(base.angulo, 2),
                "simetria": round(base.simetria, 3),
                "simetria_sin_girar": round(base.simetria_sin_girar, 3),
                "corregir": base.corregir}
        for g in GIROS:
            girada = (ndi.rotate(silueta, g, reshape=False, order=0) > 0.5).astype(np.float32)
            est = nz.estimar_giro_axial(girada)
            error = (est.angulo - base.angulo) - g
            errores[g].append(error)
            fila[f"error_{g:+d}"] = round(error, 2)
        filas.append(fila)
        print(f"{caso}  signos={diag.signos}  giro={base.angulo:+6.1f}°  "
              f"simetría {base.simetria_sin_girar:.3f}→{base.simetria:.3f}  "
              f"corregir={'sí' if base.corregir else 'no'}", flush=True)

    resumen = {f"{g:+d}°": {"error_medio_abs": round(float(np.mean(np.abs(e))), 2),
                            "error_max_abs": round(float(np.max(np.abs(e))), 2)}
               for g, e in errores.items()}
    print("\nError al recuperar giros conocidos (grados):")
    for g, r in resumen.items():
        print(f"  {g:>5}: medio {r['error_medio_abs']:.2f} · máximo {r['error_max_abs']:.2f}")

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({"casos": filas, "resumen_giros": resumen}, indent=2,
                                 ensure_ascii=False))
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
