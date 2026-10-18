// Misma paleta y textos que src/demo_render.py (Okabe-Ito, distinguible con daltonismo).
export type Region = "sacro" | "coxal_izquierdo" | "coxal_derecho";
export type RGB = [number, number, number];

export const REGION_ORDER: Region[] = ["sacro", "coxal_izquierdo", "coxal_derecho"];
export const REGION_LABELS: Record<Region, string> = {
  sacro: "Sacro",
  coxal_izquierdo: "Coxal izquierdo",
  coxal_derecho: "Coxal derecho",
};
export const REGION_STARTS: Record<Region, number> = { sacro: 1, coxal_izquierdo: 11, coxal_derecho: 21 };
export const BASE_RGB: Record<Region, RGB> = {
  sacro: [230, 159, 0],
  coxal_izquierdo: [86, 180, 233],
  coxal_derecho: [0, 158, 115],
};
export const GT_RGB: RGB = [204, 121, 167];
export const NON_CLINICAL =
  "USO EXCLUSIVAMENTE ACADÉMICO. No es un dispositivo médico, no está validado clínicamente y no debe usarse para decisiones quirúrgicas reales.";

export function regionOf(id: number): Region {
  return REGION_ORDER[Math.floor((id - 1) / 10)];
}

export function localName(id: number): string {
  const k = id - REGION_STARTS[regionOf(id)];
  return k === 0 ? "principal" : `F${k}`;
}

export function fragmentRgb(id: number): RGB {
  const region = regionOf(id);
  const k = id - REGION_STARTS[region];
  const base = BASE_RGB[region];
  if (k === 0) return [...base] as RGB;
  const tint = Math.min(0.3 + 0.12 * k, 0.8);
  return base.map((c) => Math.trunc(c + (255 - c) * tint)) as RGB;
}

export const hex = (rgb: RGB) => "#" + rgb.map((c) => c.toString(16).padStart(2, "0")).join("");
export const css = (rgb: RGB, a = 1) => `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;

export const LUT: Uint8Array = (() => {
  const lut = new Uint8Array(256 * 3);
  for (let id = 1; id <= 30; id++) lut.set(fragmentRgb(id), id * 3);
  return lut;
})();

export function distanceText(mm: number | null | undefined): string {
  if (mm === null || mm === undefined) return "sin medir";
  if (mm < 0.05) return "0.0 mm (contacto)";
  return `${mm.toFixed(1)} mm`;
}

export function shortDistance(mm: number | null | undefined): string {
  return mm === null || mm === undefined ? "sin medir" : `${mm.toFixed(1)} mm`;
}
