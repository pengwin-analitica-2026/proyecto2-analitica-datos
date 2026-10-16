"""Arma la carpeta que se sube a GitHub para Streamlit Community Cloud (``build/streamlit_cloud``).

Solo lleva el dashboard, ``demo_render.py`` y los casos ya exportados. No lleva torch ni el
checkpoint: la inferencia en vivo la hace la API de AWS (``api/server.py``), cuya dirección y clave
se configuran como secretos en Streamlit Cloud (``INFERENCE_URL`` e ``INFERENCE_KEY``).

    python3 deploy/build_streamlit_cloud.py --model h8_aumm_noise01_noema_w32_e60_s42 --cases 024 044 099 039
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIREMENTS = """\
streamlit
numpy
scipy
plotly
pillow
pandas
"""

README = """\
# Fracturas pélvicas en CT — dashboard

Proyecto del curso de Analítica de Datos (Universidad Autónoma de Occidente): detección,
segmentación por fragmentos y distancia de separación en mm sobre CT pélvico (dataset PENGWIN,
CC BY 4.0, https://doi.org/10.5281/zenodo.10990767).

**Prototipo académico. No es un dispositivo médico ni debe usarse para decisiones clínicas.**

Archivo principal en Streamlit Cloud: `app/dashboard.py`.
Secretos opcionales para la inferencia en vivo: `INFERENCE_URL` e `INFERENCE_KEY`.
Los casos mostrados son del conjunto de validación.
"""

SECRETS_EXAMPLE = """\
# Copia este contenido en Streamlit Cloud: App → Settings → Secrets
INFERENCE_URL = "https://XXXXXXXX.lambda-url.us-east-1.on.aws"
INFERENCE_KEY = "la-misma-clave-que-API_KEY-en-AWS"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out", default=str(ROOT / "build" / "streamlit_cloud"))
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "app").mkdir(parents=True)
    (out / "src").mkdir()
    (out / ".streamlit").mkdir()
    shutil.copy(ROOT / "app" / "dashboard.py", out / "app" / "dashboard.py")
    shutil.copy(ROOT / "src" / "demo_render.py", out / "src" / "demo_render.py")

    demo = ROOT / "outputs" / "demo"
    for case in args.cases:
        source = demo / args.model / case
        if not (source / "meta.json").exists():
            sys.exit(f"El caso {case} no está exportado en {source}")
        shutil.copytree(source, out / "outputs" / "demo" / args.model / case)
    if (demo / "latency.json").exists():
        shutil.copy(demo / "latency.json", out / "outputs" / "demo" / "latency.json")

    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / "README.md").write_text(README)
    (out / ".streamlit" / "secrets.toml.example").write_text(SECRETS_EXAMPLE)
    (out / ".gitignore").write_text("__pycache__/\n.streamlit/secrets.toml\n")

    biggest = max((p for p in out.rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Listo en {out} ({size:.0f} MB; archivo más grande {biggest.name}, {biggest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
