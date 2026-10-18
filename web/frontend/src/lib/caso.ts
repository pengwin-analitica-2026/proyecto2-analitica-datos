import type { Caja } from "./api";
import type { Region } from "./paleta";

export interface Fragmento {
  id: number;
  region: Region;
  is_main: boolean;
  voxels: number;
  volume_mm3: number;
  distance_mm: number | null;
  centroid_zyx_vox: [number, number, number];
  z_range: [number, number];
}

export interface Comparacion {
  region: Region;
  gt_id: number;
  gt_mm: number;
  pred_id: number | null;
  pred_mm: number | null;
  abs_error_mm: number | null;
  dice3d: number;
}

export interface Paso {
  clave: string;
  titulo: string;
  texto: string;
  z: number | null;
  forma?: number[];
}

interface ItemBinario {
  v: [number, number];
  f: [number, number];
}

export interface Meta {
  case_id: string;
  model: { checkpoint: string; parameters: number; precision?: string; tta?: boolean };
  size: number;
  n_slices: number;
  spacing_cache_xyz: [number, number, number];
  spacing_native_xyz: [number, number, number];
  tta: boolean;
  device: string;
  ms_per_slice: number;
  distance_method?: string;
  canonical?: boolean;
  flipped?: boolean;
  enderezado?: { angulo: number; corregido: boolean; giro_minimo: number } | null;
  pasos?: Paso[];
  bone_z_range: [number, number] | null;
  fragments_pred: Fragmento[];
  fragments_gt: Fragmento[] | null;
  comparison: Comparacion[] | null;
  slices: { z: number; classification: number[]; boxes_raw: Caja[]; boxes_refined: Caja[] }[];
  subido?: { fuente: string; orientacion: { accion: string }; donde: string };
  en_vivo?: { donde: string };
  paquete: {
    version: number;
    cortes: string;
    digitos: number;
    ids: string;
    gt: string | null;
    mallas: { archivo: string; items: (ItemBinario & { clave: string })[] };
    mips: Record<"axial" | "coronal" | "sagital", string> | null;
    hueso: ({ archivo: string } & Partial<Record<"crudo" | "pulido", ItemBinario>>) | null;
    pasos: Record<string, string>;
  };
}

export interface Malla {
  verts: Float32Array; // z, y, x en mm
  faces: Uint32Array;
}

export interface Caso {
  prefijo: string;
  meta: Meta;
  ids: Uint8Array;
  gt: Uint8Array | null;
  mallas: Record<string, Malla>;
  url: (archivo: string) => string;
}

async function binario(url: string): Promise<ArrayBuffer> {
  const respuesta = await fetch(url);
  if (!respuesta.ok || !respuesta.body) throw new Error(`No pude descargar ${url} (${respuesta.status})`);
  // Los .gz se guardan comprimidos tal cual: se descomprimen aquí, sin depender de cabeceras del servidor.
  return new Response(respuesta.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

function leerMallas(buffer: ArrayBuffer, items: (ItemBinario & { clave?: string })[]): Malla[] {
  return items.map((item) => ({
    verts: new Float32Array(buffer, item.v[0], item.v[1]),
    faces: new Uint32Array(buffer, item.f[0], item.f[1]),
  }));
}

export async function cargarCaso(prefijo: string, avance?: (texto: string) => void): Promise<Caso> {
  const base = "/" + prefijo.replace(/^\/|\/$/g, "") + "/";
  const url = (archivo: string) => base + archivo;
  avance?.("Leyendo el caso…");
  const respuesta = await fetch(url("caso.json"));
  if (!respuesta.ok) throw new Error(`Este caso no está disponible (${respuesta.status}).`);
  const meta = (await respuesta.json()) as Meta;
  avance?.("Descargando máscaras y mallas…");
  const p = meta.paquete;
  const [ids, gt, mallasBuf] = await Promise.all([
    binario(url(p.ids)),
    p.gt ? binario(url(p.gt)) : Promise.resolve(null),
    binario(url(p.mallas.archivo)),
  ]);
  const mallas: Record<string, Malla> = {};
  leerMallas(mallasBuf, p.mallas.items).forEach((m, i) => (mallas[p.mallas.items[i].clave] = m));
  return { prefijo, meta, ids: new Uint8Array(ids), gt: gt ? new Uint8Array(gt) : null, mallas, url };
}

export async function cargarHueso(caso: Caso): Promise<Partial<Record<"crudo" | "pulido", Malla>>> {
  const hueso = caso.meta.paquete.hueso;
  if (!hueso) return {};
  const buffer = await binario(caso.url(hueso.archivo));
  const salida: Partial<Record<"crudo" | "pulido", Malla>> = {};
  (["crudo", "pulido"] as const).forEach((v) => {
    const item = hueso[v];
    if (item) salida[v] = leerMallas(buffer, [item])[0];
  });
  return salida;
}

export function urlCorte(caso: Caso, z: number): string {
  const p = caso.meta.paquete;
  return caso.url(p.cortes.replace("{z}", String(z).padStart(p.digitos, "0")));
}

// Cortes ya decodificados (gris uint8), compartidos entre el visor y la inferencia en vivo.
const cacheCortes = new Map<string, Promise<Uint8Array>>();

export function grisDelCorte(caso: Caso, z: number): Promise<Uint8Array> {
  const url = urlCorte(caso, z);
  let promesa = cacheCortes.get(url);
  if (!promesa) {
    promesa = new Promise<Uint8Array>((resolve, reject) => {
      const img = new Image();
      img.decoding = "async";
      img.onload = () => {
        const n = caso.meta.size;
        const lienzo = document.createElement("canvas");
        lienzo.width = lienzo.height = n;
        const ctx = lienzo.getContext("2d", { willReadFrequently: true })!;
        ctx.drawImage(img, 0, 0, n, n);
        const rgba = ctx.getImageData(0, 0, n, n).data;
        const gris = new Uint8Array(n * n);
        for (let i = 0; i < gris.length; i++) gris[i] = rgba[i * 4];
        resolve(gris);
      };
      img.onerror = () => {
        cacheCortes.delete(url);
        reject(new Error(`No pude cargar el corte ${z + 1}`));
      };
      img.src = url;
    });
    cacheCortes.set(url, promesa);
    if (cacheCortes.size > 160) cacheCortes.delete(cacheCortes.keys().next().value!);
  }
  return promesa;
}
