// Visualizador 2: el mismo dibujo que render_slice de src/demo_render.py, en un canvas.
import type { Caja } from "./api";
import type { Fragmento } from "./caso";
import { BASE_RGB, GT_RGB, LUT, REGION_LABELS, css, fragmentRgb, localName, shortDistance } from "./paleta";

export interface OpcionesCorte {
  gris: Uint8Array; // [S*S]
  ids: Uint8Array; // [S*S] ya filtrado por fragmentos visibles
  gt?: Uint8Array | null;
  fragmentos: Fragmento[];
  cajas?: Caja[] | null;
  mostrarMascaras: boolean;
  mostrarCajas: boolean;
  mostrarGt: boolean;
  titulo?: string;
  escala?: number;
  alpha?: number;
}

const FUENTE = "'Inter', system-ui, sans-serif";

export function dibujarCorte(lienzo: HTMLCanvasElement, o: OpcionesCorte, size: number): void {
  const escala = o.escala ?? 2;
  const alpha = o.alpha ?? 0.45;
  const lado = size * escala;
  if (lienzo.width !== lado) lienzo.width = lienzo.height = lado;
  const ctx = lienzo.getContext("2d", { willReadFrequently: true })!;

  // Gris escalado con interpolación (como Image.BILINEAR); las máscaras van al vecino más cercano.
  const fuente = document.createElement("canvas");
  fuente.width = fuente.height = size;
  const fctx = fuente.getContext("2d")!;
  const img = fctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const g = o.gris[i];
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = g;
    img.data[i * 4 + 3] = 255;
  }
  fctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "medium";
  ctx.drawImage(fuente, 0, 0, lado, lado);

  const salida = ctx.getImageData(0, 0, lado, lado);
  const px = salida.data;
  const idEn = (arr: Uint8Array, x: number, y: number) => arr[Math.floor(y / escala) * size + Math.floor(x / escala)];

  // Contorno de una máscara = la máscara menos su erosión con cruz (borde de la imagen cuenta como fondo).
  const esContorno = (arr: Uint8Array, x: number, y: number, v: number) =>
    x === 0 || y === 0 || x === lado - 1 || y === lado - 1 ||
    idEn(arr, x - 1, y) !== v || idEn(arr, x + 1, y) !== v || idEn(arr, x, y - 1) !== v || idEn(arr, x, y + 1) !== v;

  if (o.mostrarMascaras) {
    for (let y = 0; y < lado; y++) {
      for (let x = 0; x < lado; x++) {
        const v = idEn(o.ids, x, y);
        if (!v) continue;
        const k = (y * lado + x) * 4;
        // Los secundarios llevan contorno blanco; el principal (1, 11, 21) no.
        if ((v - 1) % 10 !== 0 && esContorno(o.ids, x, y, v)) {
          px[k] = px[k + 1] = px[k + 2] = 255;
          continue;
        }
        px[k] = (1 - alpha) * px[k] + alpha * LUT[v * 3];
        px[k + 1] = (1 - alpha) * px[k + 1] + alpha * LUT[v * 3 + 1];
        px[k + 2] = (1 - alpha) * px[k + 2] + alpha * LUT[v * 3 + 2];
      }
    }
  }
  if (o.mostrarGt && o.gt) {
    for (let y = 0; y < lado; y++) {
      for (let x = 0; x < lado; x++) {
        const v = idEn(o.gt, x, y);
        if (!v || !esContorno(o.gt, x, y, v)) continue;
        const k = (y * lado + x) * 4;
        [px[k], px[k + 1], px[k + 2]] = GT_RGB;
      }
    }
  }
  ctx.putImageData(salida, 0, 0);

  const chica = Math.max(11, Math.floor(lado / 50));
  const grande = Math.max(12, Math.floor(lado / 46));
  ctx.textBaseline = "top";
  ctx.lineJoin = "round";

  if (o.mostrarCajas && o.cajas) {
    const titulos: [number, number, number, number][] = [];
    ctx.font = `700 ${chica}px ${FUENTE}`;
    for (const caja of o.cajas) {
      const [x0, y0, x1, y1] = caja.box.map((v) => v * lado);
      const rgb = BASE_RGB[caja.region];
      ctx.strokeStyle = css(rgb);
      ctx.lineWidth = Math.max(2, escala);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      const texto = `${REGION_LABELS[caja.region]} ${caja.score.toFixed(2)}`;
      const w = ctx.measureText(texto).width;
      const h = chica;
      const tx = x0 + 3;
      let ty = Math.max(y0 - h - 6, 0); // arriba de la caja...
      if (titulos.some((r) => tx < r[2] + 4 && tx + w > r[0] - 4 && ty < r[3] + 2 && ty + h > r[1] - 2))
        ty = Math.min(y1 + 4, lado - h - 2); // ...o abajo si choca con otro título
      titulos.push([tx, ty, tx + w, ty + h]);
      ctx.lineWidth = 4;
      ctx.strokeStyle = "#000";
      ctx.strokeText(texto, tx, ty);
      ctx.fillStyle = css(rgb);
      ctx.fillText(texto, tx, ty);
    }
  }

  dibujarRotulos(ctx, rotulosDelCorte(o.ids, o.fragmentos, size), escala, lado, grande);

  if (o.titulo) {
    ctx.font = `700 ${chica}px ${FUENTE}`;
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#000";
    ctx.strokeText(o.titulo, 8, 6);
    ctx.fillStyle = "#fff";
    ctx.fillText(o.titulo, 8, 6);
  }
}

interface Rotulo {
  id: number;
  texto: string;
  x: number;
  y: number;
}

/** Fragmentos secundarios presentes en el corte, con su centroide en píxeles (slice_labels). */
function rotulosDelCorte(ids: Uint8Array, fragmentos: Fragmento[], size: number): Rotulo[] {
  const porId = new Map(fragmentos.map((f) => [f.id, f]));
  const sumas = new Map<number, [number, number, number]>();
  for (let i = 0; i < ids.length; i++) {
    const v = ids[i];
    if (!v) continue;
    const s = sumas.get(v) ?? [0, 0, 0];
    s[0] += i % size;
    s[1] += Math.floor(i / size);
    s[2] += 1;
    sumas.set(v, s);
  }
  const rotulos: Rotulo[] = [];
  for (const [id, [sx, sy, n]] of [...sumas.entries()].sort((a, b) => a[0] - b[0])) {
    const f = porId.get(id);
    if (!f || f.is_main || n < 4) continue;
    rotulos.push({ id, texto: `${localName(id)} · ${shortDistance(f.distance_mm)}${f.metal ? " ⚠" : ""}`, x: sx / n, y: sy / n });
  }
  return rotulos;
}

function dibujarRotulos(ctx: CanvasRenderingContext2D, rotulos: Rotulo[], escala: number, lado: number, fuente: number) {
  const pad = 4;
  const gap = 3;
  const puestos: [number, number, number, number][] = [];
  ctx.font = `700 ${fuente}px ${FUENTE}`;
  for (const r of [...rotulos].sort((a, b) => a.y - b.y || a.x - b.x)) {
    const w = ctx.measureText(r.texto).width + 2 * pad + 4;
    const h = fuente + 2 * pad;
    const cx = r.x * escala;
    const cy = r.y * escala;
    const x0 = Math.min(Math.max(cx - w / 2, 2), lado - w - 2);
    let y0 = Math.min(Math.max(cy - h - 10, 2), lado - h - 2); // justo encima del fragmento
    let movido = true;
    while (movido) {
      // baja el rótulo hasta que no choque
      movido = false;
      for (const [px0, py0, px1, py1] of puestos) {
        if (x0 < px1 + gap && x0 + w > px0 - gap && y0 < py1 + gap && y0 + h > py0 - gap) {
          y0 = py1 + gap;
          movido = true;
        }
      }
    }
    y0 = Math.min(y0, lado - h - 2);
    puestos.push([x0, y0, x0 + w, y0 + h]);
    const acento = css(fragmentRgb(r.id));
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x0 + w / 2, y0 + h);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = acento;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "rgba(10,10,10,0.8)";
    ctx.beginPath();
    ctx.roundRect(x0, y0, w, h, 5);
    ctx.fill();
    ctx.fillStyle = acento;
    ctx.fillRect(x0, y0 + 3, 4, h - 6);
    ctx.fillStyle = "#fff";
    ctx.fillText(r.texto, x0 + pad + 4, y0 + pad);
  }
}
