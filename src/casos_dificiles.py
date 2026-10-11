"""Herramientas del notebook 05: mirar casos difíciles y probar qué pasa si se reorientan.

Todo es diagnóstico: no entrena ni cambia umbrales. Los casos de test (012, 039) se abren solo
para entender el fallo; ninguna decisión se toma con ellos (ver docs/decisiones_semana10.md,
sección 12).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pengwin_io  # noqa: E402
from analisis_instancias import as3d, match_slice, predict_case  # noqa: E402
from build_case_cache import build_case, load_case_cache  # noqa: E402
from evaluate_volumes import PRIMARY_IDS, apply_stress, load_model, semantic_from_labels  # noqa: E402
from instance_postprocess import primary_secondary_to_instances  # noqa: E402

REGIONES = ("fondo", "sacro", "coxal izq.", "coxal der.")


# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
def leer_cabecera(case_id: str) -> dict:
    """Lee la cabecera de texto del .mha: orientación, matriz de dirección, tamaño y espaciado."""
    path = pengwin_io.find_image_path(case_id)
    if path is None:            # sin data/raw (por ejemplo en Colab): se usa la lista ya leída
        from orientacion import is_rotated
        girado = is_rotated(case_id)
        return {"caso": str(case_id).zfill(3), "orientacion": "LPI" if girado else "RAI",
                "matriz_direccion": "-1 0 0 0 -1 0 0 0 1" if girado else "1 0 0 0 1 0 0 0 1",
                "tamano_xyz": None, "espaciado_mm": None, "guardado_girado_180": girado,
                "fuente": "lista de src/orientacion.py"}
    header = {}
    with open(path, "rb") as stream:
        for line in stream.read(4096).split(b"\n"):
            try:
                text = line.decode("ascii")
            except UnicodeDecodeError:
                break
            if "=" not in text:
                break
            key, value = (part.strip() for part in text.split("=", 1))
            header[key] = value
            if key == "ElementDataFile":
                break
    return {"caso": str(case_id).zfill(3), "orientacion": header.get("AnatomicalOrientation"),
            "matriz_direccion": header.get("TransformMatrix"), "tamano_xyz": header.get("DimSize"),
            "espaciado_mm": header.get("ElementSpacing"),
            "guardado_girado_180": header.get("TransformMatrix", "").startswith("-1 0 0 0 -1"),
            "fuente": "cabecera del .mha"}


def cargar_caso(case_id: str, cache_root: str | Path = "data/cache/s10", size: int = 384) -> dict:
    """Devuelve images/labels [Z,S,S] del caché; si falta, lo construye desde data/raw."""
    case_id = str(case_id).zfill(3)
    cache_dir = Path(cache_root) / f"cache_{size}"
    if not (cache_dir / f"{case_id}.npz").exists():
        print(f"construyendo caché del caso {case_id} (unos segundos)...")
        build_case(case_id, str(cache_root), (size,), pengwin_io.GAMMA, "diagnostico")
    return load_case_cache(cache_dir, case_id)


def cargar_modelo(checkpoint: str | Path, device: str = "cpu", tta: bool = True):
    dev = torch.device(device)
    model, _ = load_model([str(checkpoint)], dev, tta)
    return model, dev


# ----------------------------------------------------------------------------------------------
# Predicción y métricas
# ----------------------------------------------------------------------------------------------
def predecir(model, images: np.ndarray, device) -> dict:
    sem, ps, _ = predict_case(model, images, device)
    return {"semantica": sem, "instancias": as3d(primary_secondary_to_instances(ps, sem))}


def metricas(pred: dict, labels: np.ndarray) -> dict:
    """Resumen del caso y Dice de instancias por corte."""
    sem, inst = pred["semantica"], pred["instancias"]
    gt = semantic_from_labels(labels)
    conf = np.bincount(gt.ravel().astype(np.int64) * 4 + sem.ravel().astype(np.int64), minlength=16).reshape(4, 4)
    bone = max(conf[1:].sum(), 1)
    por_corte, filas = [], []
    for z in np.flatnonzero((labels != 0).any(axis=(1, 2))):
        rows, _ = match_slice(inst[z], labels[z])
        filas += rows
        por_corte.append((int(z), float(np.mean([r[1] for r in rows]))))
    resumen = {f"dice_{REGIONES[c]}": round(float(2 * conf[c, c] / max(conf[c].sum() + conf[:, c].sum(), 1)), 3)
               for c in (1, 2, 3)}
    resumen.update({
        "hueso_tomado_por_otra_region_%": round(100 * float((conf[1:, 1:].sum() - np.trace(conf[1:, 1:])) / bone), 1),
        "hueso_predicho_como_fondo_%": round(100 * float(conf[1:, 0].sum() / bone), 1),
        "dice_instancias": round(float(np.mean([r[1] for r in filas])), 3) if filas else float("nan"),
        "dice_primario": round(float(np.mean([r[1] for r in filas if r[0] in PRIMARY_IDS])), 3) if filas else float("nan"),
    })
    secundarios = [r[1] for r in filas if r[0] not in PRIMARY_IDS]
    resumen["dice_secundario"] = round(float(np.mean(secundarios)), 3) if secundarios else float("nan")
    return {"resumen": resumen, "por_corte": por_corte, "confusion": conf}


def girar(images: np.ndarray, labels: np.ndarray, grados: float) -> tuple[np.ndarray, np.ndarray]:
    """Gira imagen y etiqueta juntas (imagen bilineal, etiqueta vecino más cercano)."""
    if abs(grados) < 1e-6:
        return images, labels
    return apply_stress(images, labels, f"rot={grados}")


def angulo_del_sacro(semantica: np.ndarray) -> float:
    """Hacia dónde queda el sacro respecto a los coxales, en grados (0 = abajo en la imagen).

    En la orientación habitual (paciente boca arriba, anterior arriba) el sacro queda detrás
    de los coxales, es decir abajo: ángulo cercano a 0. Un volumen guardado girado da ~180 y
    un paciente de lado, un valor intermedio. Solo usa sacro contra coxales, no izquierda/derecha.
    """
    sacro = np.argwhere(semantica == 1)
    coxales = np.argwhere(semantica >= 2)
    if len(sacro) < 50 or len(coxales) < 50:
        return float("nan")
    dy = sacro[:, 1].mean() - coxales[:, 1].mean()
    dx = sacro[:, 2].mean() - coxales[:, 2].mean()
    return math.degrees(math.atan2(dx, dy))


def estimar_giro(semantica: np.ndarray) -> float:
    """Ángulo (grados) que hay que girar el volumen para dejar el sacro abajo."""
    actual = angulo_del_sacro(semantica)
    if not np.isfinite(actual):
        return 0.0
    mejor, residuo = 0.0, abs(actual)
    for candidato in (actual, -actual):          # el signo depende de la convención de la rejilla
        _, girada = apply_stress(semantica[::4], semantica[::4], f"rot={candidato}")
        nuevo = angulo_del_sacro(girada)
        if np.isfinite(nuevo) and abs(nuevo) < residuo:
            mejor, residuo = candidato, abs(nuevo)
    return float(mejor)


def barrido_de_giros(model, device, images: np.ndarray, labels: np.ndarray,
                     angulos=(-135, -90, -45, 0, 45, 90, 135, 180), paso: int = 3) -> list[dict]:
    """Dice del caso si se girara ``angulo`` grados antes de entrar al modelo (1 de cada ``paso`` cortes)."""
    keep = np.flatnonzero((labels != 0).any(axis=(1, 2)))[::paso]
    out = []
    for angulo in angulos:
        im, lb = girar(images[keep], labels[keep], angulo)
        resumen = metricas(predecir(model, im, device), lb)["resumen"]
        out.append({"giro": angulo, **resumen})
        print(f"giro {angulo:>5}°: instancias {resumen['dice_instancias']:.3f} | "
              f"otra región {resumen['hueso_tomado_por_otra_region_%']:.1f} %", flush=True)
    return out


# ----------------------------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------------------------
def _colores():
    """Un color por región (azul sacro, rojo coxal izq., verde coxal der.); secundarios más claros."""
    from matplotlib.colors import ListedColormap
    base = {0: (0.12, 0.47, 0.90), 1: (0.90, 0.20, 0.20), 2: (0.15, 0.70, 0.30)}
    extra = [(1.0, 0.75, 0.0), (0.75, 0.3, 0.95), (0.0, 0.85, 0.9), (1.0, 0.45, 0.75), (0.6, 0.6, 0.1),
             (0.95, 0.55, 0.2), (0.5, 0.5, 1.0), (0.4, 0.9, 0.6), (0.8, 0.8, 0.8)]
    table = [(0, 0, 0)]
    for region in range(3):
        table.append(base[region])
        table.extend(extra)
    return ListedColormap(table[:31])


def mostrar(images: np.ndarray, labels: np.ndarray, pred: dict, z: int, titulo: str = "", ax=None):
    """CT, etiqueta real y predicción de un corte. Azul sacro, rojo coxal izq., verde coxal der."""
    import matplotlib.pyplot as plt
    cmap = _colores()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2)) if ax is None else (None, ax)
    rows, _ = match_slice(pred["instancias"][z], labels[z]) if (labels[z] > 0).any() else ([], 0)
    dice = f" | Dice instancias del corte {np.mean([r[1] for r in rows]):.2f}" if rows else ""
    for a, data, name in zip(axes, (None, labels[z], pred["instancias"][z]), ("CT", "Real", "Predicción")):
        a.imshow(images[z], cmap="gray", vmin=0, vmax=255)
        if data is not None:
            a.imshow(np.ma.masked_equal(data, 0), cmap=cmap, vmin=0, vmax=30, alpha=0.55, interpolation="nearest")
            name += f"  ids {[int(v) for v in np.unique(data) if v > 0]}"
        a.set_title(name, fontsize=10)
        a.axis("off")
    if fig is not None:
        fig.suptitle(f"{titulo} corte {z}{dice}", fontsize=11)
        fig.tight_layout()
    return fig


def mosaico(images: np.ndarray, labels: np.ndarray, pred: dict, titulo: str = "", n: int = 5):
    """``n`` cortes repartidos a lo largo del hueso."""
    bone = np.flatnonzero((labels != 0).any(axis=(1, 2)))
    for z in bone[np.linspace(0, len(bone) - 1, n + 2).astype(int)[1:-1]]:
        mostrar(images, labels, pred, int(z), titulo)


def curva_por_corte(resultados: dict[str, dict]):
    """Dice de instancias a lo largo del volumen para varios casos."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for nombre, m in resultados.items():
        z = np.asarray([p[0] for p in m["por_corte"]], dtype=float)
        ax.plot((z - z.min()) / max(z.max() - z.min(), 1), [p[1] for p in m["por_corte"]], label=nombre, lw=1.2)
    ax.axhline(0.85, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("posición a lo largo del hueso (0 = primer corte con hueso)")
    ax.set_ylabel("Dice de instancias del corte")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    return fig
