/**
 * Colour maps. Each is a list of RGB stops sampled evenly on [0, 1]; lookup builds a 256-entry table once.
 *
 * DIVERGING is a blue-grey-red cool/warm map (Moreland), readable by people with red-green colour
 * blindness because it varies in lightness as well as hue. SEQUENTIAL is viridis (perceptually
 * uniform, colour-blind safe) for rain and spread. GREY is for the infrared, cold cloud tops light.
 */

export const DIVERGING = [[59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249], [221, 221, 221], [245, 196, 173], [244, 154, 123], [222, 96, 77], [180, 4, 38]];
export const SEQUENTIAL = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];
export const GREY = [[20, 24, 30], [240, 242, 245]];
export const MASK = [120, 126, 134];

const tables = new Map();

function build(stops) {
  const t = new Uint8ClampedArray(256 * 3);
  const n = stops.length - 1;
  for (let i = 0; i < 256; i++) {
    const p = (i / 255) * n, j = Math.min(n - 1, Math.floor(p)), f = p - j;
    for (let c = 0; c < 3; c++) t[i * 3 + c] = stops[j][c] + (stops[j + 1][c] - stops[j][c]) * f;
  }
  return t;
}

/** @returns {Uint8ClampedArray} 256 x RGB lookup for the named map */
export function lookup(stops) {
  if (!tables.has(stops)) tables.set(stops, build(stops));
  return tables.get(stops);
}

/** CSS gradient string for a legend swatch. */
export function gradient(stops, reverse = false) {
  const s = reverse ? [...stops].reverse() : stops;
  return `linear-gradient(90deg, ${s.map(c => `rgb(${c.join(',')})`).join(',')})`;
}
