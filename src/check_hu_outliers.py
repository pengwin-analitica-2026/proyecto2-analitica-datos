"""
check_hu_outliers.py
---------------------
Lee data/manifest.csv (generado por build_manifest.py) e identifica los
casos con valores HU fuera del rango típico de un CT (-1000 a ~4000 HU),
para decidir si son artefactos de metal, padding, o un problema real.

Uso:
    python check_hu_outliers.py
"""

import pandas as pd

MANIFEST_PATH = "data/manifest.csv"

# Rango "esperado" para un CT normal (con margen para metal de fijación)
HU_MIN_ESPERADO = -1100
HU_MAX_ESPERADO = 5000

df = pd.read_csv(MANIFEST_PATH)
validos = df[df["valido"]].copy()

print(f"Total casos válidos: {len(validos)}")
print(f"\nTop 5 casos con HU máximo más alto:")
print(validos.nlargest(5, "hu_max")[["case_id", "hu_min", "hu_max", "hu_mean"]].to_string(index=False))

print(f"\nTop 5 casos con HU mínimo más bajo:")
print(validos.nsmallest(5, "hu_min")[["case_id", "hu_min", "hu_max", "hu_mean"]].to_string(index=False))

fuera_de_rango = validos[
    (validos["hu_min"] < HU_MIN_ESPERADO) | (validos["hu_max"] > HU_MAX_ESPERADO)
]

print(f"\nCasos con valores fuera del rango esperado ({HU_MIN_ESPERADO} a {HU_MAX_ESPERADO} HU): {len(fuera_de_rango)}")
if len(fuera_de_rango) > 0:
    print(fuera_de_rango[["case_id", "hu_min", "hu_max", "hu_mean"]].to_string(index=False))
