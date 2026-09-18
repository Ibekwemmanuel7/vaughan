/**
 * Canvas rendering of 2-D fields.
 *
 * A FieldView owns one visible canvas. The field is colour-mapped once into an offscreen canvas at its
 * native resolution, then drawn onto the visible canvas at device-pixel size with bilinear smoothing, so
 * the picture is crisp on high-density screens and a resize costs one drawImage rather than a remap.
 * Every view carries role="img" and an aria-label that names the field and its range.
 */
import { DIVERGING, GREY, MASK, SEQUENTIAL, lookup } from './colormap.js';

const KIND = { div: DIVERGING, seq: SEQUENTIAL, grey: GREY };

export class FieldView {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.off = document.createElement('canvas');
    this.offCtx = this.off.getContext('2d');
    this.dirty = false;
    canvas.setAttribute('role', 'img');
    this.ro = new ResizeObserver(() => this.blit());
    this.ro.observe(canvas);
  }

  /**
   * Colour-map a field into the offscreen buffer and repaint.
   * @param {{data:Float32Array,h:number,w:number}} f
   * @param {'div'|'seq'|'grey'} kind
   * @param {number} vmin  @param {number} vmax
   * @param {{invert?:boolean, maskBelow?:number, label?:string}} [opt]
   */
  draw(f, kind, vmin, vmax, opt = {}) {
    const { h, w, data } = f;
    if (this.off.width !== w || this.off.height !== h) { this.off.width = w; this.off.height = h; }
    const img = this.offCtx.createImageData(w, h);
    const lut = lookup(KIND[kind]);
    const px = img.data;
    const scale = 255 / (vmax - vmin);
    const maskBelow = opt.maskBelow ?? -Infinity;
    for (let i = 0; i < w * h; i++) {
      const v = data[i];
      const k = i * 4;
      if (!(v > maskBelow) || !Number.isFinite(v)) { px[k] = MASK[0]; px[k + 1] = MASK[1]; px[k + 2] = MASK[2]; px[k + 3] = 255; continue; }
      let t = (v - vmin) * scale;
      if (opt.invert) t = 255 - t;
      const j = (t < 0 ? 0 : t > 255 ? 255 : t | 0) * 3;
      px[k] = lut[j]; px[k + 1] = lut[j + 1]; px[k + 2] = lut[j + 2]; px[k + 3] = 255;
    }
    this.offCtx.putImageData(img, 0, 0);
    if (opt.label) this.canvas.setAttribute('aria-label', opt.label);
    this.blit();
  }

  /** Copy the offscreen buffer to the visible canvas at device-pixel size. */
  blit() {
    if (!this.off.width) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const r = this.canvas.getBoundingClientRect();
    const W = Math.max(1, Math.round(r.width * dpr)), H = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
    const c = this.ctx;
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = 'high';
    c.clearRect(0, 0, W, H);
    c.drawImage(this.off, 0, 0, W, H);
  }

  destroy() { this.ro.disconnect(); }
}

/** Create a FieldView for every canvas in a root element, keyed by id. */
export function mountFields(root) {
  const views = new Map();
  for (const c of root.querySelectorAll('canvas.field')) views.set(c.id, new FieldView(c));
  return views;
}
