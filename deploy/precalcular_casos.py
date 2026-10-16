"""Precalcula los casos de PENGWIN para el dashboard y los sube a S3 (``exportados/<modelo>/<caso>/``).

    AWS_PROFILE=almia python deploy/precalcular_casos.py --checkpoint outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth \\
        --procesos 4 [--casos 001 044] [--sin-tta]

Cada caso pasa por ``demo_export.export_case`` (el mismo pipeline de la demo, con los pasos del
preprocesamiento) y se sube con ``almacen_s3.subir_exportado``. Corre ``--procesos`` procesos en
paralelo, cada uno con ``--hilos`` hilos de torch: casi la mitad del tiempo por caso es CPU
(distancias, mallas, vista cruda), así que repartir los casos rinde más que una GPU sola. Los casos que
ya están en S3 se saltan: se puede cortar y volver a correr. Necesita ``data/raw`` con los 100 casos.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def trabajar(casos: list[str], checkpoint: str, tta: bool, salida: Path) -> None:
    import almacen_s3
    from demo_export import export_case
    modelo = Path(checkpoint).stem
    ya = set(almacen_s3._prefijos(f"exportados/{modelo}/"))
    for caso in casos:
        if caso in ya:
            print(f"{caso}: ya estaba en S3", flush=True)
            continue
        t = time.time()
        carpeta = export_case(caso, checkpoint, out_root=str(salida), tta=tta, progress=lambda m: None)
        almacen_s3.subir_exportado(carpeta, modelo, caso)
        print(f"{caso}: listo y subido en {time.time() - t:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs/s10/h8_aumm_noise01_noema_w32_e60_s42.pth"))
    parser.add_argument("--casos", nargs="*", default=[f"{i:03d}" for i in range(1, 101)])
    parser.add_argument("--procesos", type=int, default=4)
    parser.add_argument("--hilos", type=int, default=2)
    parser.add_argument("--sin-tta", action="store_true")
    parser.add_argument("--salida", default=None, help="Carpeta local de trabajo (por defecto, una temporal).")
    parser.add_argument("--trabajador", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    salida = Path(args.salida or Path(tempfile.gettempdir()) / "pengwin_precalculo")
    if args.trabajador:
        import torch
        torch.set_num_threads(args.hilos)
        trabajar(args.casos, args.checkpoint, not args.sin_tta, salida)
        return
    grupos = [args.casos[i::args.procesos] for i in range(args.procesos)]
    entorno = dict(os.environ, OMP_NUM_THREADS=str(args.hilos), MKL_NUM_THREADS=str(args.hilos))
    procesos = []
    for n, grupo in enumerate(g for g in grupos if g):
        comando = [sys.executable, "-u", __file__, "--trabajador", "--checkpoint", args.checkpoint,
                   "--hilos", str(args.hilos), "--salida", str(salida), "--casos", *grupo]
        if args.sin_tta:
            comando.append("--sin-tta")
        procesos.append(subprocess.Popen(comando, env=entorno))
        print(f"proceso {n + 1}: {len(grupo)} casos ({grupo[0]}…{grupo[-1]})", flush=True)
    codigos = [p.wait() for p in procesos]
    print("terminado" if not any(codigos) else f"algún proceso falló: {codigos}", flush=True)
    sys.exit(max(codigos))


if __name__ == "__main__":
    main()
