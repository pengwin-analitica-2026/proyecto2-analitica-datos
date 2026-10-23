import type { Region } from "./paleta";

// En AWS la API es la Function URL de la Lambda pelvis-web; en desarrollo, el proxy de Vite.
const API = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export interface Destino {
  id: "cpu" | "gpu";
  nombre: string;
  detalle: string;
}

export interface GpuEstado {
  endpoint: string | null;
  estado: string | null;
  apagado_programado: string | null;
  horas_apagado?: number;
}

export interface Config {
  modelo: string;
  destinos: Destino[];
  gpu: GpuEstado | null;
  max_subida_mb: number;
  almacen: string;
}

export interface CasoPengwin {
  caso: string;
  split: string;
  girado: boolean;
  crudo: boolean;
  precalculado: boolean;
  listo_web: boolean;
}

export interface Casos {
  modelo: string;
  ejemplos: { caso: string; prefijo: string }[];
  pengwin: CasoPengwin[];
  error: string | null;
}

export interface Trabajo {
  id: string;
  tipo: string;
  estado: "en_cola" | "procesando" | "listo" | "error";
  log: { t: number; m: string }[];
  resultado: string | null;
  error: string | null;
  creado: number;
  actualizado: number;
}

export interface Caja {
  region_id: number;
  region: Region;
  score: number;
  box: [number, number, number, number];
}

export interface CorteVivo {
  semantica_b64: string;
  boxes_refined: Caja[];
  boxes_raw: Caja[];
  latency_ms: number;
  device: string;
  model: string;
  precision: string;
  donde: string;
  servidor_ms: number;
}

async function pedir<T>(ruta: string, init?: RequestInit): Promise<T> {
  let respuesta: Response;
  try {
    respuesta = await fetch(API + ruta, init);
  } catch {
    throw new Error("No hay conexión con el servidor.");
  }
  if (!respuesta.ok) {
    let detalle = `${respuesta.status}`;
    try {
      const cuerpo = await respuesta.json();
      detalle = typeof cuerpo.detail === "string" ? cuerpo.detail : JSON.stringify(cuerpo.detail ?? cuerpo);
    } catch {
      /* respuesta sin JSON */
    }
    throw new Error(detalle);
  }
  return respuesta.json() as Promise<T>;
}

const json = (cuerpo: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(cuerpo),
});

export const api = {
  config: () => pedir<Config>("/api/config"),
  casos: () => pedir<Casos>("/api/casos"),
  abrir: (cuerpo: { caso: string; modo: "precalculado" | "vivo"; destino: string; tta: boolean }) =>
    pedir<{ resultado?: string; trabajo?: string }>("/api/abrir", json(cuerpo)),
  trabajo: (id: string) => pedir<Trabajo>(`/api/trabajos/${id}`),
  nuevaSubida: (archivos: { nombre: string; tamano: number }[]) =>
    pedir<{ id: string; formularios: { url: string; fields: Record<string, string> }[] }>(
      "/api/subidas",
      json({ archivos }),
    ),
  procesarSubida: (id: string, cuerpo: { destino: string; tta: boolean; enderezar: boolean; grosor_mm: number | null; reducir_metal: boolean }) =>
    pedir<{ trabajo: string }>(`/api/subidas/${id}/procesar`, json(cuerpo)),
  corte: (bytes: Uint8Array, destino: string) =>
    pedir<CorteVivo>("/api/corte", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Destino": destino },
      body: bytes,
    }),
  gpu: () => pedir<GpuEstado>("/api/gpu"),
  gpuAccion: (accion: "encender" | "apagar", clave: string) =>
    pedir<GpuEstado>(`/api/gpu/${accion}`, { method: "POST", headers: { "X-Admin-Key": clave } }),
};

/** Sube un archivo con el formulario firmado de S3 (o el de desarrollo), informando el avance en bytes. */
export function subirArchivo(
  formulario: { url: string; fields: Record<string, string> },
  archivo: File,
  avance: (bytes: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const datos = new FormData();
    Object.entries(formulario.fields).forEach(([k, v]) => datos.append(k, v));
    datos.append("file", archivo);
    const xhr = new XMLHttpRequest();
    const url = formulario.url.startsWith("/") ? API + formulario.url : formulario.url;
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => avance(e.loaded);
    xhr.onload = () =>
      xhr.status < 300 ? resolve() : reject(new Error(`La subida de ${archivo.name} falló (${xhr.status}).`));
    xhr.onerror = () => reject(new Error(`Se cortó la conexión subiendo ${archivo.name}.`));
    xhr.send(datos);
  });
}
