"""Tablas y figuras de la Semana 10 a partir de los resultados guardados en ``outputs/s10``.

No entrena ni infiere: solo lee los JSON que dejaron ``train_s10.py``, ``evaluate_volumes.py``
e ``infer_volume.py``. Lo usa el notebook ``04_semana10_pipeline_multitarea.ipynb``; por eso no
importa ``torch`` y corre en cualquier equipo con pandas y matplotlib.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
S10 = RAIZ / "outputs" / "s10"

OBJETIVOS = {
    "classification_f1_macro": ("F1 de clasificación (macro)", 0.85),
    "classification_auc_macro": ("AUC de clasificación (macro)", 0.85),
    "detection_iou_mean": ("IoU medio de detección (caja refinada)", 0.65),
    "detection_map50": ("mAP50 de detección (refinada)", 0.65),
    "detection_map50_95": ("mAP50:95 de detección (refinada)", 0.40),
    "segmentation_dice_macro": ("Dice de segmentación semántica (macro)", 0.85),
    "segmentation_iou_macro": ("IoU de segmentación semántica (macro)", 0.70),
    "instance_dice_matched": ("Dice por fragmento", 0.85),
    "instance_iou_matched": ("IoU por fragmento", 0.70),
}

COLUMNAS = {
    "classification_f1_macro": "F1",
    "detection_map50": "mAP50",
    "detection_map50_95": "mAP50:95",
    "detection_raw_map50": "mAP50 cruda",
    "segmentation_dice_macro": "Dice sem.",
    "instance_dice_matched": "Dice inst.",
    "instance_dice_primary": "Primario",
    "instance_dice_secondary": "Secundario",
    "instance_missed_secondary": "Sec. sin pareja",
}

# Paleta (validada con el validador de la guía de visualización; el orden es fijo).
SUPERFICIE, TINTA, TINTA_2, TENUE, REJILLA, EJE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RAMPA_AZUL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def existe(nombre: str) -> bool:
    return (S10 / f"{nombre}.json").exists()


def cargar(nombre: str) -> dict:
    """Lee ``outputs/s10/<nombre>.json``."""
    return json.loads((S10 / f"{nombre}.json").read_text())


def resumen(nombre: str, subconjunto: str = "todo") -> dict:
    """Métricas de un JSON de ``evaluate_volumes.py`` para un subconjunto de cortes."""
    return cargar(nombre)["subsets"][subconjunto]


def tabla_modelos(filas: list[tuple[str, str]], columnas: dict[str, str] | None = None,
                  decimales: int = 3) -> pd.DataFrame:
    """Una fila por modelo. ``filas`` es una lista de (etiqueta, nombre del JSON de evaluación)."""
    columnas = columnas or COLUMNAS
    datos = []
    for etiqueta, nombre in filas:
        if not existe(nombre):
            datos.append({"Modelo": etiqueta, **{titulo: None for titulo in columnas.values()}})
            continue
        metricas = resumen(nombre)
        datos.append({"Modelo": etiqueta, **{titulo: metricas.get(clave) for clave, titulo in columnas.items()}})
    return pd.DataFrame(datos).set_index("Modelo").round(decimales)


def tabla_objetivos(evaluaciones: list[tuple[str, str]]) -> pd.DataFrame:
    """Las nueve métricas del enunciado contra su objetivo, una columna por evaluación."""
    datos = []
    for clave, (titulo, objetivo) in OBJETIVOS.items():
        fila = {"Métrica": titulo, "Objetivo": objetivo}
        for etiqueta, nombre in evaluaciones:
            valor = resumen(nombre)[clave]
            fila[etiqueta] = round(valor, 3)
            fila[f"{etiqueta} cumple"] = "sí" if valor >= objetivo else "NO"
        datos.append(fila)
    return pd.DataFrame(datos).set_index("Métrica")


def tabla_por_caso(evaluaciones: list[tuple[str, str]], clave: str = "instance_dice_matched") -> pd.DataFrame:
    """Una fila por caso y una columna por evaluación."""
    columnas = {}
    for etiqueta, nombre in evaluaciones:
        columnas[etiqueta] = {fila["case_id"]: fila[clave] for fila in cargar(nombre)["per_case"]}
    return pd.DataFrame(columnas).sort_index().round(3)


def tabla_distancias(evaluaciones: list[tuple[str, str]]) -> pd.DataFrame:
    """Resumen de ``infer_volume.py``: distancia en mm y Dice 3D."""
    claves = {
        "error_distancia_mm_media": "Error medio (mm)",
        "error_distancia_mm_p90": "Error p90 (mm)",
        "correlacion_distancia": "Correlación",
        "n_fragmentos_secundarios": "Secundarios",
        "n_con_distancia_predicha": "Con distancia predicha",
        "dice_principal": "Dice 3D principal",
        "dice_secundario": "Dice 3D secundario",
        "perdidos_secundario": "Secundarios perdidos",
        "assd_mm_media": "ASSD (mm)",
        "hd95_mm_media": "HD95 (mm)",
    }
    datos = {}
    for etiqueta, nombre in evaluaciones:
        res = cargar(nombre)["resumen"]
        datos[etiqueta] = {titulo: res.get(clave) for clave, titulo in claves.items()}
    return pd.DataFrame(datos).round(3)


def fragmentos_con_distancia(nombre: str) -> pd.DataFrame:
    """Una fila por fragmento secundario con su distancia real y predicha (mm)."""
    filas = []
    for caso in cargar(nombre)["casos"]:
        for frag in caso["distancias"]:
            filas.append({"caso": caso["case_id"], "región": frag["region"],
                          "real (mm)": frag.get("ground_truth_mm"), "predicha (mm)": frag.get("predicted_mm"),
                          "error (mm)": frag.get("absolute_error_mm"), "Dice 3D": frag.get("dice")})
    return pd.DataFrame(filas).round(2)


def historia(nombre: str) -> pd.DataFrame:
    """Historial por época de una corrida de ``train_s10.py``."""
    return pd.DataFrame(cargar(nombre)["history"])


# ----------------------------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------------------------
def _ejes(ancho: float = 8.0, alto: float = 3.6):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(ancho, alto), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(EJE)
    ax.tick_params(colors=TENUE, labelsize=9, length=0)
    ax.grid(axis="both", color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def figura_progresion(filas: list[tuple[str, str]], objetivo: float = 0.85):
    """Dice por fragmento en validación a lo largo de los modelos que marcaron un cambio."""
    fig, ax = _ejes(8.0, 0.42 * len(filas) + 1.2)
    etiquetas = [etiqueta for etiqueta, _ in filas][::-1]
    valores = [resumen(nombre)["instance_dice_matched"] for _, nombre in filas][::-1]
    # Puntos y no barras: el eje no empieza en cero, y una barra recortada exagera las diferencias.
    ax.scatter(valores, etiquetas, s=70, color=SERIES[0], edgecolor=SUPERFICIE, linewidth=1.5, zorder=3)
    ax.axvline(objetivo, color=TINTA_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(objetivo + 0.0008, len(filas) - 1, f"objetivo {objetivo:.2f}", color=TINTA_2, fontsize=9, ha="left", va="center")
    for y, valor in enumerate(valores):
        ax.text(valor + 0.0012, y + 0.28, f"{valor:.3f}", va="center", ha="left", fontsize=9, color=TINTA)
    ax.set_xlim(min(valores) - 0.01, max(max(valores), objetivo) + 0.012)
    ax.set_ylim(-0.6, len(filas) - 0.3)
    ax.tick_params(axis="y", labelcolor=TINTA, labelsize=9.5)
    ax.set_xlabel("Dice por fragmento en validación (15 casos, TTA)", color=TINTA_2, fontsize=9.5)
    ax.set_title("Qué cambios movieron la métrica", loc="left", fontsize=11.5, color=TINTA)
    fig.tight_layout()
    return fig


def figura_estres(series: list[tuple[str, list[tuple[float, str]]]]):
    """Dice por fragmento según el giro aplicado a la validación. ``series``: (modelo, [(grados, JSON)])."""
    fig, ax = _ejes(7.6, 4.0)
    finales = []
    for indice, (modelo, puntos) in enumerate(series):
        xs = [grados for grados, _ in puntos]
        ys = [resumen(nombre)["instance_dice_matched"] for _, nombre in puntos]
        ax.plot(xs, ys, color=SERIES[indice], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SUPERFICIE, markeredgewidth=1.5, label=modelo)
        finales.append([ys[-1], xs[-1], f"{modelo}  {ys[-1]:.2f}"])
    finales.sort()                                   # etiquetas de abajo hacia arriba, sin encimarse
    for anterior, actual in zip(finales, finales[1:]):
        actual.append(max(actual[0], anterior[-1] + 0.062 if len(anterior) > 3 else anterior[0] + 0.062))
    finales[0].append(finales[0][0])
    for valor, x, texto, y_texto in ((f[0], f[1], f[2], f[-1]) for f in finales):
        ax.text(x + 1.2, y_texto, texto, va="center", ha="left", fontsize=9.5, color=TINTA)
    ax.set_xticks([0, 30, 60])
    ax.set_xticklabels(["sin girar", "30°", "60°"])
    ax.set_xlim(-3, 80)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice por fragmento", color=TINTA_2, fontsize=9.5)
    ax.set_xlabel("Giro aplicado a los volúmenes de validación", color=TINTA_2, fontsize=9.5)
    ax.set_title("Tolerancia al giro según la rotación vista al entrenar", loc="left", fontsize=11.5, color=TINTA)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=TINTA_2)
    fig.tight_layout()
    return fig


def figura_por_caso(antes: tuple[str, str], despues: tuple[str, str], objetivo: float = 0.85):
    """Dice por fragmento de cada caso con dos modelos, unidos por una línea."""
    tabla = tabla_por_caso([antes, despues]).sort_values(despues[0])
    fig, ax = _ejes(7.6, 0.33 * len(tabla) + 1.4)
    ys = range(len(tabla))
    for y, (_, fila) in zip(ys, tabla.iterrows()):
        ax.plot([fila[antes[0]], fila[despues[0]]], [y, y], color=EJE, linewidth=1.5, zorder=1)
    ax.scatter(tabla[antes[0]], ys, s=54, color=SERIES[1], edgecolor=SUPERFICIE, linewidth=1.5, zorder=2, label=antes[0])
    ax.scatter(tabla[despues[0]], ys, s=54, color=SERIES[0], edgecolor=SUPERFICIE, linewidth=1.5, zorder=3, label=despues[0])
    ax.axvline(objetivo, color=TINTA_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"caso {caso}" for caso in tabla.index])
    ax.tick_params(axis="y", labelcolor=TINTA, labelsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Dice por fragmento del caso (línea punteada: objetivo {objetivo:.2f})", color=TINTA_2, fontsize=9.5)
    ax.set_title("Cada caso de test, antes y después", loc="left", fontsize=11.5, color=TINTA)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TINTA_2)
    fig.tight_layout()
    return fig


def figura_matriz(matriz, etiquetas: list[str], titulo: str, pie: str = ""):
    """Matriz de confusión normalizada por fila (porcentaje de cada clase real)."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    matriz = np.asarray(matriz, dtype=float)
    porcentaje = 100 * matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    fig, ax = _ejes(6.0, 4.8)
    ax.grid(visible=False)
    ax.imshow(porcentaje, cmap=LinearSegmentedColormap.from_list("azul", RAMPA_AZUL), vmin=0, vmax=100)
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            valor = porcentaje[i, j]
            texto = f"{valor:.1f} %" if valor >= 0.05 else "0"
            ax.text(j, i, texto, ha="center", va="center", fontsize=9.5, color="#ffffff" if valor > 55 else TINTA)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels([etiqueta.replace(" ", "\n") for etiqueta in etiquetas])
    ax.set_yticklabels(etiquetas)
    ax.tick_params(colors=TINTA, labelsize=9.5)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Predicho" + (f"\n{pie}" if pie else ""), color=TINTA_2, fontsize=9.5)
    ax.set_ylabel("Real", color=TINTA_2, fontsize=9.5)
    ax.set_title(titulo, loc="left", fontsize=11.5, color=TINTA)
    fig.tight_layout()
    return fig
