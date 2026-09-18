/**
 * Data access for the dashboard.
 *
 * The build packs every 2-D field into one little-endian Uint16 buffer, linearly quantised per array
 * (value = lo + q * (hi - lo) / 65535), and writes the index into manifest.json. Decoding is lazy and
 * cached, so a field costs nothing until a panel asks for it.
 *
 * @typedef {{lo:number, hi:number, off:number, shape:[number, number]}} FieldEntry
 * @typedef {{data:Float32Array, h:number, w:number}} Field
 */

const cache = new Map();

/** Fetch JSON with a clear error message. */
export async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

/** Fetch the field buffer; returns a decoder bound to the manifest index. */
export async function loadFields(manifest) {
  const r = await fetch(manifest.fields.file);
  if (!r.ok) throw new Error(`${manifest.fields.file}: HTTP ${r.status}`);
  const u16 = new Uint16Array(await r.arrayBuffer());
  const index = manifest.fields.index;
  return {
    /** @returns {Field} */
    get(key) {
      if (cache.has(key)) return cache.get(key);
      const e = index[key];
      if (!e) throw new Error(`no field ${key}`);
      const [h, w] = e.shape;
      const n = h * w;
      const out = new Float32Array(n);
      const k = (e.hi - e.lo) / 65535;
      for (let i = 0; i < n; i++) out[i] = e.lo + u16[e.off + i] * k;
      const f = { data: out, h, w };
      cache.set(key, f);
      return f;
    },
    has(key) { return key in index; },
  };
}

/** Field minus its mean (finite values only). */
export function anomaly(f) {
  let s = 0, n = 0;
  for (const v of f.data) if (Number.isFinite(v)) { s += v; n++; }
  const m = n ? s / n : 0;
  const data = new Float32Array(f.data.length);
  for (let i = 0; i < data.length; i++) data[i] = f.data[i] - m;
  return { data, h: f.h, w: f.w };
}

/** Field minus the mean of each row (cross-sections: anomaly from the level mean). */
export function rowAnomaly(f) {
  const data = new Float32Array(f.data.length);
  for (let y = 0; y < f.h; y++) {
    let s = 0, n = 0;
    for (let x = 0; x < f.w; x++) { const v = f.data[y * f.w + x]; if (Number.isFinite(v)) { s += v; n++; } }
    const m = n ? s / n : 0;
    for (let x = 0; x < f.w; x++) data[y * f.w + x] = f.data[y * f.w + x] - m;
  }
  return { data, h: f.h, w: f.w };
}

/** Add a constant. */
export function shifted(f, c) {
  const data = new Float32Array(f.data.length);
  for (let i = 0; i < data.length; i++) data[i] = f.data[i] + c;
  return { data, h: f.h, w: f.w };
}

/** Min and max of the values above a floor (masked pixels are stored as 0). */
export function rangeAbove(f, floor) {
  let lo = Infinity, hi = -Infinity;
  for (const v of f.data) if (v > floor) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return lo === Infinity ? null : [lo, hi];
}
