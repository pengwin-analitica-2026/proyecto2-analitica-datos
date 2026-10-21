"""Dibujo de los visualizadores 2 y 3 y de la vista cruda (Semana 11).

Solo dibuja a partir de lo que dejó ``demo_export``; no carga el modelo.

- ``render_slice``: visualizador 2. Corte en gris con las máscaras coloreadas por macro-hueso, las
  cajas de detección y, junto a cada fragmento secundario, su ID y la distancia en mm al fragmento
  principal. La distancia es del fragmento entero en 3D, no del corte; se repite en cada corte
  donde el fragmento aparece.
- ``mesh_figure``: visualizador 3. Una malla Plotly ``Mesh3d`` por fragmento con la misma paleta, y
  la etiqueta ``ID · mm`` en el centroide de cada fragmento secundario.
- ``bone_figure``: isosuperficie del hueso crudo (visualizador 1).

Paleta (Okabe-Ito, distinguible con daltonismo): sacro naranja, coxal izquierdo celeste, coxal
derecho verde azulado. El fragmento principal usa el tono pleno; los secundarios, tintes más claros
del mismo tono, con contorno blanco en el corte.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

REGION_ORDER = ("sacro", "coxal_izquierdo", "coxal_derecho")
REGION_LABELS = {"sacro": "Sacro", "coxal_izquierdo": "Coxal izquierdo", "coxal_derecho": "Coxal derecho"}
REGION_STARTS = {"sacro": 1, "coxal_izquierdo": 11, "coxal_derecho": 21}
BASE_RGB = {"sacro": (230, 159, 0), "coxal_izquierdo": (86, 180, 233), "coxal_derecho": (0, 158, 115)}
GT_RGB = (204, 121, 167)
NON_CLINICAL = ("USO EXCLUSIVAMENTE ACADÉMICO. No es un dispositivo médico, no está validado "
                "clínicamente y no debe usarse para decisiones quirúrgicas reales.")


def region_of(fragment_id: int) -> str:
    return REGION_ORDER[(int(fragment_id) - 1) // 10]


def local_name(fragment_id: int) -> str:
    """'principal' para 1/11/21 y 'F1', 'F2'... para los secundarios de la región."""
    k = int(fragment_id) - REGION_STARTS[region_of(fragment_id)]
    return "principal" if k == 0 else f"F{k}"


def fragment_rgb(fragment_id: int) -> tuple[int, int, int]:
    """Tono pleno del macro-hueso para el principal; tinte más claro por cada secundario."""
    region = region_of(fragment_id)
    k = int(fragment_id) - REGION_STARTS[region]
    base = np.asarray(BASE_RGB[region], dtype=float)
    if k == 0:
        return tuple(int(c) for c in base)
    tint = min(0.30 + 0.12 * k, 0.80)
    return tuple(int(c + (255 - c) * tint) for c in base)


def hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for fragment_id in range(1, 31):
        lut[fragment_id] = fragment_rgb(fragment_id)
    return lut


_LUT = _lut()
_FONTS: dict[int, ImageFont.ImageFont] = {}


def _font(size: int) -> ImageFont.ImageFont:
    if size not in _FONTS:
        try:
            import matplotlib
            path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
            _FONTS[size] = ImageFont.truetype(str(path), size)
        except Exception:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def _contour(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask)


def distance_text(distance_mm: float | None) -> str:
    if distance_mm is None:
        return "sin medir"
    if distance_mm < 0.05:
        return "0.0 mm (contacto)"
    return f"{distance_mm:.1f} mm"


def short_distance(distance_mm: float | None) -> str:
    """Texto breve para rótulos junto al fragmento: siempre en mm, nunca en píxeles."""
    return "sin medir" if distance_mm is None else f"{distance_mm:.1f} mm"


def slice_labels(ids_slice: np.ndarray, fragments: list[dict]) -> list[dict]:
    """Etiquetas de los fragmentos secundarios que aparecen en este corte: texto y posición (px)."""
    by_id = {row["id"]: row for row in fragments}
    labels = []
    for fragment_id in (int(v) for v in np.unique(ids_slice) if v > 0):
        row = by_id.get(fragment_id)
        if row is None or row["is_main"]:
            continue
        ys, xs = np.nonzero(ids_slice == fragment_id)
        if len(xs) < 4:
            continue
        labels.append({"id": fragment_id, "text": f"{local_name(fragment_id)} · {short_distance(row['distance_mm'])}"
                       + (" ⚠" if row.get("metal") else ""),
                       "x": float(xs.mean()), "y": float(ys.mean()), "distance_mm": row["distance_mm"],
                       "region": row["region"]})
    return labels


def render_slice(image_u8: np.ndarray, ids_slice: np.ndarray, fragments: list[dict],
                 slice_meta: dict | None = None, gt_slice: np.ndarray | None = None,
                 boxes: str = "refined", show_masks: bool = True, show_boxes: bool = True,
                 show_gt: bool = False, scale: int = 2, alpha: float = 0.45,
                 title: str | None = None) -> Image.Image:
    """Visualizador 2 para un corte. ``boxes`` es 'refined' (con la semántica) o 'raw' (cabeza grid)."""
    size = image_u8.shape[0]
    side = size * scale
    gray = Image.fromarray(image_u8).resize((side, side), Image.BILINEAR)
    canvas = np.asarray(gray.convert("RGB")).astype(np.float32)
    ids_big = np.repeat(np.repeat(ids_slice, scale, axis=0), scale, axis=1)

    if show_masks:
        mask = ids_big > 0
        color = _LUT[ids_big].astype(np.float32)
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color[mask]
        for fragment_id in (int(v) for v in np.unique(ids_slice) if v > 0):
            if local_name(fragment_id) == "principal":
                continue
            edge = _contour(ids_big == fragment_id)
            canvas[edge] = (255, 255, 255)
    if show_gt and gt_slice is not None:
        gt_big = np.repeat(np.repeat(gt_slice, scale, axis=0), scale, axis=1)
        for fragment_id in (int(v) for v in np.unique(gt_slice) if v > 0):
            canvas[_contour(gt_big == fragment_id)] = GT_RGB

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    small, big = _font(max(11, side // 50)), _font(max(12, side // 46))

    if show_boxes and slice_meta:
        tags: list[tuple[float, float, float, float]] = []
        for box in slice_meta["boxes_refined" if boxes == "refined" else "boxes_raw"]:
            x0, y0, x1, y1 = (v * side for v in box["box"])
            rgb = BASE_RGB[box["region"]]
            draw.rectangle([x0, y0, x1, y1], outline=rgb + (255,), width=max(2, scale))
            tag = f"{REGION_LABELS[box['region']]} {box['score']:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), tag, font=small)
            w, h = right - left, bottom - top
            tx, ty = x0 + 3, max(y0 - h - 6, 0)                    # arriba de la caja...
            if any(tx < r[2] + 4 and tx + w > r[0] - 4 and ty < r[3] + 2 and ty + h > r[1] - 2 for r in tags):
                ty = min(y1 + 4, side - h - 2)                       # ...o abajo si choca con otro título
            tags.append((tx, ty, tx + w, ty + h))
            draw.text((tx, ty - top), tag, fill=rgb + (255,), font=small, stroke_width=2, stroke_fill=(0, 0, 0, 255))

    _draw_fragment_labels(draw, slice_labels(ids_slice, fragments), scale, side, big)

    if title:
        draw.text((8, 6), title, fill=(255, 255, 255, 255), font=small, stroke_width=2,
                  stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(image, layer).convert("RGB")


def _draw_fragment_labels(draw: ImageDraw.ImageDraw, labels: list[dict], scale: int, side: int,
                          font: ImageFont.ImageFont) -> None:
    """Rótulos ``F2 · 19.0 mm`` con fondo, sin pisarse entre sí y con una línea al fragmento."""
    pad, gap = 4, 3
    placed: list[tuple[float, float, float, float]] = []
    for label in sorted(labels, key=lambda l: (l["y"], l["x"])):
        left, top, right, bottom = draw.textbbox((0, 0), label["text"], font=font)
        w, h = right - left + 2 * pad, bottom - top + 2 * pad
        cx, cy = label["x"] * scale, label["y"] * scale
        x0 = min(max(cx - w / 2, 2), side - w - 2)
        y0 = min(max(cy - h - 10, 2), side - h - 2)               # justo encima del fragmento
        moved = True
        while moved:                                               # baja el rótulo hasta que no choque
            moved = False
            for px0, py0, px1, py1 in placed:
                if x0 < px1 + gap and x0 + w > px0 - gap and y0 < py1 + gap and y0 + h > py0 - gap:
                    y0 = py1 + gap
                    moved = True
        y0 = min(y0, side - h - 2)
        placed.append((x0, y0, x0 + w, y0 + h))
        accent = fragment_rgb(label["id"])
        draw.line([(cx, cy), (x0 + w / 2, y0 + h)], fill=(255, 255, 255, 230), width=2)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=accent + (255,), outline=(255, 255, 255, 255), width=2)
        draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=5, fill=(10, 10, 10, 205))
        draw.rectangle([x0, y0 + 3, x0 + 4, y0 + h - 3], fill=accent + (255,))
        draw.text((x0 + pad + 4, y0 + pad - top), label["text"], fill=(255, 255, 255, 255), font=font)


def legend_items(fragments: list[dict]) -> list[tuple[str, str]]:
    """(hex, texto) de la leyenda: un color por macro-hueso presente."""
    present = {row["region"] for row in fragments}
    return [(hex_color(BASE_RGB[r]), REGION_LABELS[r]) for r in REGION_ORDER if r in present]


def mesh_figure(meshes: dict[str, np.ndarray], fragments: list[dict],
                spacing_xyz: tuple[float, float, float], show_main: bool = True,
                main_opacity: float = 0.40, show_gt: bool = False,
                gt_fragments: list[dict] | None = None, title: str | None = None,
                label_contacts: bool = True):
    """Visualizador 3: una malla por fragmento, con ID y distancia junto al fragmento.

    ``label_contacts=False`` deja rótulo solo a los fragmentos separados del principal (> 0 mm).
    """
    import plotly.graph_objects as go
    sx, sy, sz = spacing_xyz
    spacing_zyx = np.array([sz, sy, sx])
    fig = go.Figure()

    def add_mesh(prefix: str, row: dict, rgb, opacity: float, name: str):
        verts, faces = meshes.get(f"{prefix}_{row['id']}_v"), meshes.get(f"{prefix}_{row['id']}_f")
        if verts is None:
            return
        fig.add_trace(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=hex_color(rgb), opacity=opacity, name=name, showlegend=False, flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6),
            hovertemplate=name + "<extra></extra>"))

    for row in fragments:
        is_main = row["is_main"]
        if is_main and not show_main:
            continue
        region = REGION_LABELS[row["region"]]
        dist = "" if is_main else f" · {short_distance(row['distance_mm'])}"
        add_mesh("p", row, fragment_rgb(row["id"]), main_opacity if is_main else 1.0,
                 f"{region} · {local_name(row['id'])}{dist}")
    if show_gt and gt_fragments:
        for row in gt_fragments:
            add_mesh("g", row, GT_RGB, 0.18, f"Real · {REGION_LABELS[row['region']]} · {local_name(row['id'])}")

    secondary = [r for r in fragments if not r["is_main"]]
    if secondary:
        centers = np.array([np.asarray(r["centroid_zyx_vox"]) * spacing_zyx for r in secondary])
        fig.add_trace(go.Scatter3d(
            x=centers[:, 2], y=centers[:, 1], z=centers[:, 0], mode="markers",
            marker=dict(size=4, color="#111111"), showlegend=False,
            hovertext=[f"{REGION_LABELS[r['region']]} · {local_name(r['id'])} · {distance_text(r['distance_mm'])}"
                       for r in secondary], hoverinfo="text"))
    annotations = []
    for index, row in enumerate(r for r in secondary if label_contacts or (r["distance_mm"] or 0) >= 0.05):
        z, y, x = np.asarray(row["centroid_zyx_vox"]) * spacing_zyx
        annotations.append(dict(
            x=float(x), y=float(y), z=float(z), text=f"<b>{local_name(row['id'])} · {short_distance(row['distance_mm'])}"
                 f"{' ⚠' if row.get('metal') else ''}</b>",
            showarrow=True, arrowhead=0, arrowwidth=1.5, arrowcolor="#222222",
            ax=-46 + 28 * (index % 3), ay=-48 - 16 * (index % 4),          # reparte los rótulos para no apilarlos
            font=dict(size=13, color="#111111"), bgcolor="rgba(255,255,255,0.88)",
            bordercolor=hex_color(fragment_rgb(row["id"])), borderwidth=2, borderpad=3))
    fig.update_layout(
        title=dict(text=title or "", font=dict(size=14)),
        scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
                   annotations=annotations,
                   camera=dict(eye=dict(x=0.0, y=-2.1, z=0.55), up=dict(x=0, y=0, z=1))),   # vista anterior, cabeza arriba
        margin=dict(l=0, r=0, t=40 if title else 0, b=0), height=640,
        uirevision="keep")
    return fig


def bone_figure(verts_zyx: np.ndarray, faces: np.ndarray, title: str | None = None):
    """Isosuperficie del hueso crudo (visualizador 1)."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Mesh3d(
        x=verts_zyx[:, 2], y=verts_zyx[:, 1], z=verts_zyx[:, 0], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="ivory", opacity=1.0, flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5)))
    fig.update_layout(title=dict(text=title or "", font=dict(size=14)),
                      scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)",
                                 zaxis_title="z (mm)"),
                      margin=dict(l=0, r=0, t=40 if title else 0, b=0), height=560)
    return fig
