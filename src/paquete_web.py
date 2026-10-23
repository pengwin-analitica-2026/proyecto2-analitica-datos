"""Paquete web de un caso exportado: lo que lee la página de AWS (``web/``) en el navegador.

``demo_export`` deja ``meta.json`` y varios ``.npz``, pensados para Python. El navegador no lee ``.npz``
con comodidad y no conviene bajarle 200 MB de golpe, así que cada caso se convierte una vez en:

    caso.json          meta.json + índice del paquete («paquete»); se escribe AL FINAL: si existe, está completo
    cortes/0000.png    cada corte uint8 384×384 tal como entra a la red (PNG sin pérdida: la inferencia en
                       vivo de un corte manda esos mismos píxeles al modelo). Se bajan a medida que se ven.
    ids.bin.gz         volumen de IDs PENGWIN predicho, uint8 [z,S,S], gzip (casi todo es fondo: pesa poco)
    gt.bin.gz          ground truth con la misma forma, si hay
    mallas.bin.gz      mallas por fragmento (vértices float32 z,y,x en mm y caras uint32), una tras otra
    hueso.bin.gz       isosuperficie del hueso crudo y pulido (visualizador 1)
    mip_*.png          MIP axial, coronal y sagital (visualizador 1)
    pasos/*.png        preprocesamiento paso a paso

El índice de ``mallas.bin.gz`` y ``hueso.bin.gz`` (desplazamiento y cantidad de cada arreglo, en bytes) va
en ``caso.json``. ``resumen_proyecto`` arma ``datos/resumen.json`` con las métricas de test y la latencia
que muestra la pestaña «Métricas y latencia».
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = 1


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def _png_hu(corte_hu: np.ndarray) -> bytes:
    """Corte en HU como PNG RGB sin pérdida: (HU + 1024) en 16 bits, byte alto en R y bajo en G.

    El navegador lee un PNG de 8 bits por canal sin tocar los valores; uno de 16 bits por canal no se puede
    leer en un canvas sin perder precisión. Así el visor recupera el HU exacto: R·256 + G − 1024.
    """
    valor = (np.clip(corte_hu.astype(np.int32), -1024, 64511) + 1024).astype(np.uint16)
    rgb = np.zeros(valor.shape + (3,), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1] = valor >> 8, valor & 0xFF
    return _png(rgb)


def _gz(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=6, mtime=0)


class _Binario:
    """Concatena arreglos de 4 bytes por elemento y anota dónde quedó cada uno."""

    def __init__(self) -> None:
        self.partes: list[bytes] = []
        self.offset = 0

    def agregar(self, array: np.ndarray, dtype) -> list[int]:
        data = np.ascontiguousarray(array, dtype=dtype).tobytes()
        entrada = [self.offset, int(np.asarray(array).size)]
        self.partes.append(data)
        self.offset += len(data)
        return entrada

    def bytes(self) -> bytes:
        return b"".join(self.partes)


def empaquetar(carpeta: str | Path, escribir) -> dict:
    """Convierte la carpeta de ``demo_export`` en el paquete web.

    ``escribir(nombre, datos)`` guarda cada archivo (en disco o en S3). Devuelve el contenido de
    ``caso.json``, que se escribe último.
    """
    carpeta = Path(carpeta)
    meta = json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))
    paquete: dict = {"version": VERSION, "cortes": "cortes/{z}.png", "digitos": 4}
    with np.load(carpeta / "volume.npz") as volumen:
        images = volumen["images"]
        for z in range(len(images)):
            escribir(f"cortes/{z:04d}.png", _png(images[z]))
        del images
        paquete["hu"] = None
        if "hu" in volumen.files:                 # exportaciones nuevas: ventana de grises de rango completo
            hu = volumen["hu"]
            for z in range(len(hu)):
                escribir(f"hu/{z:04d}.png", _png_hu(hu[z]))
            paquete["hu"] = "hu/{z}.png"
            del hu
        escribir("ids.bin.gz", _gz(np.ascontiguousarray(volumen["ids"], dtype=np.uint8).tobytes()))
        paquete["ids"] = "ids.bin.gz"
        paquete["gt"] = None
        if "gt_ids" in volumen.files:
            escribir("gt.bin.gz", _gz(np.ascontiguousarray(volumen["gt_ids"], dtype=np.uint8).tobytes()))
            paquete["gt"] = "gt.bin.gz"

    binario, items = _Binario(), []
    with np.load(carpeta / "meshes.npz") as mallas:
        for clave in sorted(k[:-2] for k in mallas.files if k.endswith("_v")):
            if f"{clave}_f" not in mallas.files:
                continue
            items.append({"clave": clave, "v": binario.agregar(mallas[f"{clave}_v"], np.float32),
                          "f": binario.agregar(mallas[f"{clave}_f"], np.uint32)})
    escribir("mallas.bin.gz", _gz(binario.bytes()))
    paquete["mallas"] = {"archivo": "mallas.bin.gz", "items": items}

    paquete["mips"] = paquete["hueso"] = None
    crudo = carpeta / "raw_view.npz"
    if crudo.exists():
        with np.load(crudo) as vista:
            paquete["mips"] = {}
            for eje in ("axial", "coronal", "sagital"):
                escribir(f"mip_{eje}.png", _png(vista[f"mip_{eje}"]))
                paquete["mips"][eje] = f"mip_{eje}.png"
            binario, hueso = _Binario(), {}
            for version, sufijo in (("crudo", ""), ("pulido", "_pulido")):
                if f"bone_verts_zyx{sufijo}" in vista.files and f"bone_faces{sufijo}" in vista.files:
                    hueso[version] = {"v": binario.agregar(vista[f"bone_verts_zyx{sufijo}"], np.float32),
                                      "f": binario.agregar(vista[f"bone_faces{sufijo}"], np.uint32)}
            if hueso:
                escribir("hueso.bin.gz", _gz(binario.bytes()))
                paquete["hueso"] = dict(hueso, archivo="hueso.bin.gz")

    paquete["pasos"] = {}
    pasos = carpeta / "pasos.npz"
    if pasos.exists() and meta.get("pasos"):
        with np.load(pasos) as arreglos:
            for paso in meta["pasos"]:
                for vista in ("corte", "mip"):
                    nombre = f"{paso['clave']}_{vista}"
                    if nombre in arreglos.files:
                        escribir(f"pasos/{nombre}.png", _png(arreglos[nombre]))
                        paquete["pasos"][nombre] = f"pasos/{nombre}.png"

    caso = dict(meta, paquete=paquete)
    escribir("caso.json", json.dumps(caso).encode("utf-8"))
    return caso


def escritor_local(destino: str | Path):
    """``escribir`` para ``empaquetar`` que guarda en una carpeta."""
    destino = Path(destino)

    def escribir(nombre: str, datos: bytes) -> None:
        ruta = destino / nombre
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(datos)
    return escribir


# --------------------------------------------------------------------------- pestaña de métricas
TEST_ROWS = (("F1 de clasificación", "classification_f1_macro", 0.85),
             ("AUC de clasificación", "classification_auc_macro", 0.85),
             ("IoU medio de detección", "detection_iou_mean", 0.65),
             ("mAP50", "detection_map50", 0.65),
             ("mAP50:95", "detection_map50_95", 0.40),
             ("Dice de segmentación semántica", "segmentation_dice_macro", 0.85),
             ("IoU de segmentación semántica", "segmentation_iou_macro", 0.70),
             ("Dice por fragmento", "instance_dice_matched", 0.85),
             ("IoU por fragmento", "instance_iou_matched", 0.70))
TEST_EVALUATIONS = (("1.ª · D5", "s10/TEST_d5_s42_tta.json"),
                    ("2.ª · H6", "s10/TEST2_h6_s42_tta.json"),
                    ("3.ª · H7", "h7/test_tercera_h7_auggeo_rot35_canon_aumm_noema_w32_e60_s42_sin_enderezar.json"),
                    ("4.ª · H8", "h8/test/test_h8.json"))


def _leer(raiz: Path, nombre: str) -> dict | list | None:
    ruta = raiz / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def resumen_proyecto(resultados: str | Path, demo: str | Path) -> dict:
    """Lo mismo que ``test_metrics_section`` y ``latency_section`` del dashboard, como datos."""
    resultados, demo = Path(resultados), Path(demo)
    found = [(label, _leer(resultados, name)) for label, name in TEST_EVALUATIONS]
    found = [(label, data["subsets"]["todo"]) for label, data in found if data]
    final = dict(found).get("4.ª · H8")
    salida: dict = {"test": None, "indicadores": [], "latencia": []}
    if final:
        filas = []
        for name, key, goal in TEST_ROWS:
            filas.append({"metrica": name, "objetivo": goal, "valores": {label: v[key] for label, v in found},
                          "cumple": final[key] >= goal})
        salida["test"] = {"n_casos": final["n_cases"], "n_cortes": final["n_slices"],
                          "columnas": [label for label, _ in found], "filas": filas}
        surface = _leer(resultados, "h8/test/superficie_test_h8.json")
        if surface:
            r = surface["resumen"]["todos"]
            salida["indicadores"].append({
                "titulo": "HD95 / ASSD por fragmento en test (H8)",
                "valor": f"{r['hd95_mm_mediana']:.1f} / {r['assd_mm_mediana']:.2f} mm",
                "ayuda": f"Medianas sobre {r['n_detectados']} de {r['n_fragmentos']} fragmentos reales detectados "
                         f"(cobertura {r['cobertura']:.1%}) · medias {r['hd95_mm_media']:.1f} / "
                         f"{r['assd_mm_media']:.2f} mm"})
        distances, who = _leer(resultados, "h8/test/distancias_test_h8.json"), "H8"
        if not distances:
            distances, who = _leer(resultados, "s10/TEST2_h6_s42_nofill.json"), "H6"
        if distances:
            r = distances["resumen"]
            salida["indicadores"].append({
                "titulo": f"Error de distancia en test ({who})", "valor": f"{r['error_distancia_mm_media']:.2f} mm",
                "ayuda": f"{r['n_con_distancia_predicha']} de {r['n_fragmentos_secundarios']} fragmentos secundarios "
                         f"con distancia predicha · mediana {r['error_distancia_mm_mediana']:.1f} mm · "
                         f"correlación {r['correlacion_distancia']:.2f}"})
        sam, who = _leer(resultados, "h8/test/sam_baseline_test_h8.json"), "H8"
        if not sam:
            sam, who = _leer(resultados, "s10/sam_baseline_test_h6.json"), "H6"
        if sam:
            salida["indicadores"].append({
                "titulo": f"Dice por región en test: {who} vs SAM zero-shot",
                "valor": f"{sam['global_nuestro']['dice']:.3f} vs {sam['global_sam']['dice']:.3f}",
                "ayuda": f"{sam['global_sam']['n']} regiones en {sam['cortes']} cortes (uno de cada {sam['stride']}). "
                         f"SAM recibe como indicación la caja que predice {who}; no forma parte del pipeline."})

    for name in ("s10/latencia_gpu_L4.json", "s10/latencia_cpu_colab.json"):
        data = _leer(resultados, name)
        if not data:
            continue
        for config, values in data["config"].items():
            salida["latencia"].append({"dispositivo": data["device"], "configuracion": config,
                                       "mediana_ms": values["mediana_ms"], "alto": f"p95 · {values['p95_ms']}",
                                       "fuente": "Semana 10, Colab (H6, misma arquitectura)"})
    medida = _leer(demo, "latency.json")
    for row in medida or []:
        hilos = f", {row['Hilos CPU']} hilos" if row.get("Hilos CPU") not in (None, "—") else ""
        salida["latencia"].append({"dispositivo": f"{row['Dispositivo'].lower()} ({row['Hardware']}{hilos})",
                                   "configuracion": "1 modelo" + (" + TTA" if row["TTA"] == "sí" else ""),
                                   "mediana_ms": row["Mediana (ms)"], "alto": f"p90 · {row['p90 (ms)']}",
                                   "fuente": "measure_latency.py"})
    return salida


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Empaqueta casos exportados para la página web.")
    parser.add_argument("carpetas", nargs="*", help="carpetas de demo_export (outputs/demo/<modelo>/<caso>)")
    parser.add_argument("--destino", required=True, help="carpeta donde queda <destino>/<caso>/")
    parser.add_argument("--resumen", help="escribe también el resumen de métricas en este archivo")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for carpeta in args.carpetas:
        destino = Path(args.destino) / Path(carpeta).name
        caso = empaquetar(carpeta, escritor_local(destino))
        print(f"{carpeta} → {destino} ({caso['n_slices']} cortes)")
    if args.resumen:
        Path(args.resumen).parent.mkdir(parents=True, exist_ok=True)
        Path(args.resumen).write_text(json.dumps(resumen_proyecto(root / "outputs", root / "outputs" / "demo")))


if __name__ == "__main__":
    main()
