/**
 * SVG charts: the per-scene warm-core profile and the 16-scene time series.
 * Built with DOM APIs (no innerHTML), with a hover tooltip and a keyboard-reachable description.
 */

const NS = 'http://www.w3.org/2000/svg';
const isNum = v => typeof v === 'number' && Number.isFinite(v);

function el(name, attrs = {}, text) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

function polyline(points, stroke, dashed) {
  const d = points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
  const p = el('path', { d, fill: 'none', stroke, 'stroke-width': 2.2 });
  if (dashed) p.setAttribute('stroke-dasharray', '6 4');
  return p;
}

function dots(points, fill) {
  const g = el('g');
  for (const [x, y] of points) g.appendChild(el('circle', { cx: x.toFixed(1), cy: y.toFixed(1), r: 3.5, fill, stroke: 'var(--surface)', 'stroke-width': 1.5 }));
  return g;
}

/** Attach a mouse tooltip to an svg. fn(mx) returns lines or null; leave() clears any cursor. */
export function hover(svg, tip, fn, leave) {
  const move = e => {
    const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    const mx = (e.clientX - r.left) / r.width * vb.width;
    const lines = fn(mx);
    if (!lines) { tip.style.display = 'none'; return; }
    tip.replaceChildren(...lines.flatMap((l, i) => i ? [document.createElement('br'), document.createTextNode(l)] : [document.createTextNode(l)]));
    tip.style.display = 'block';
    const pr = tip.parentElement.getBoundingClientRect();
    tip.style.left = `${Math.min(e.clientX - pr.left + 14, pr.width - tip.offsetWidth - 8)}px`;
    tip.style.top = `${e.clientY - pr.top - 10}px`;
  };
  svg.addEventListener('mousemove', move);
  svg.addEventListener('mouseleave', () => { tip.style.display = 'none'; leave?.(); });
}

/**
 * Warm-core profile: anomaly (K) on x, pressure (log) on y.
 * @param {SVGSVGElement} svg
 * @param {number[]} levels
 * @param {{era5?:number[], phys?:number[], unet?:number[], ana:number[]}} series
 */
export function drawProfile(svg, levels, series) {
  const W = 560, H = 300, L = 58, R = 16, T = 14, B = 34;
  const ys = p => T + (Math.log(p) - Math.log(200)) / (Math.log(1000) - Math.log(200)) * (H - T - B);
  const xmin = -1, xmax = 5, xs = v => L + (v - xmin) / (xmax - xmin) * (W - L - R);
  clear(svg);
  const g = el('g');
  for (const p of levels) {
    g.appendChild(el('line', { x1: L, x2: W - R, y1: ys(p), y2: ys(p), stroke: 'var(--rule)' }));
    g.appendChild(el('text', { x: L - 8, y: ys(p) + 4, 'text-anchor': 'end', fill: 'var(--ink-3)' }, p));
  }
  for (let v = xmin; v <= xmax; v++) {
    g.appendChild(el('line', { y1: T, y2: H - B, x1: xs(v), x2: xs(v), stroke: v === 0 ? 'var(--ink-3)' : 'var(--rule)', 'stroke-dasharray': v === 0 ? '' : '2 3' }));
    g.appendChild(el('text', { x: xs(v), y: H - B + 16, 'text-anchor': 'middle', fill: 'var(--ink-3)' }, v));
  }
  g.appendChild(el('text', { x: (L + W - R) / 2, y: H - 4, 'text-anchor': 'middle', fill: 'var(--ink-2)' }, 'warm-core anomaly, K'));
  g.appendChild(el('text', { transform: `translate(12 ${(T + H - B) / 2}) rotate(-90)`, 'text-anchor': 'middle', fill: 'var(--ink-2)' }, 'hPa'));
  svg.appendChild(g);
  const line = (arr, col, dashed) => {
    if (!arr) return;
    const pts = arr.map((v, i) => [xs(v), ys(levels[i])]);
    svg.appendChild(polyline(pts, col, dashed)); svg.appendChild(dots(pts, col));
  };
  line(series.era5, 'var(--s-era)', true); line(series.phys, 'var(--s-phys)'); line(series.unet, 'var(--s-unet)'); line(series.ana, 'var(--s-ana)');
  const i300 = levels.indexOf(300);
  svg.setAttribute('aria-label', `Warm-core profile: analysis ${series.ana[i300].toFixed(1)} K at 300 hPa` + (series.era5 ? `, ERA5 ${series.era5[i300].toFixed(1)} K` : ''));
}

/**
 * Time series of the 300 hPa warm core across scenes.
 * @param {SVGSVGElement} svg
 * @param {HTMLElement} tip
 * @param {{labels:string[], atms:boolean[], series:{name:string, values:(number|null)[], color:string, dashed?:boolean}[], notes:[number,string][]}} spec
 */
export function drawSeries(svg, tip, spec) {
  const W = 1040, H = 320, L = 48, R = 34, T = 18, B = 44, n = spec.labels.length;
  const xs = i => L + i / (n - 1) * (W - L - R);
  const ymin = 0, ymax = 5, ys = v => T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B);
  clear(svg);
  const g = el('g');
  for (let v = ymin; v <= ymax; v++) {
    g.appendChild(el('line', { x1: L, x2: W - R, y1: ys(v), y2: ys(v), stroke: 'var(--rule)' }));
    g.appendChild(el('text', { x: L - 8, y: ys(v) + 4, 'text-anchor': 'end', fill: 'var(--ink-3)' }, v));
  }
  spec.labels.forEach((lab, i) => {
    g.appendChild(el('text', { x: xs(i), y: H - B + 16, 'text-anchor': 'middle', fill: 'var(--ink-3)', 'font-size': 10.5 }, lab));
    if (spec.atms[i]) g.appendChild(el('circle', { cx: xs(i), cy: H - B + 28, r: 3, fill: 'var(--ink-3)' }));
  });
  g.appendChild(el('text', { transform: `translate(12 ${(T + H - B) / 2}) rotate(-90)`, 'text-anchor': 'middle', fill: 'var(--ink-2)' }, 'K at 300 hPa'));
  svg.appendChild(g);
  for (const s of spec.series) {
    let run = [];
    const flush = () => { if (run.length) { svg.appendChild(polyline(run, s.color, s.dashed)); run = []; } };
    s.values.forEach((v, i) => { if (isNum(v)) run.push([xs(i), ys(v)]); else flush(); });
    flush();
    svg.appendChild(dots(s.values.map((v, i) => isNum(v) ? [xs(i), ys(v)] : null).filter(Boolean), s.color));
  }
  for (const [i, text] of spec.notes) svg.appendChild(el('text', { x: xs(i), y: T + 10, fill: 'var(--ink-2)', 'font-size': 11, 'text-anchor': 'middle' }, text));
  const cross = el('line', { x1: 0, x2: 0, y1: T, y2: H - B, stroke: 'var(--ink-3)', 'stroke-width': 1, opacity: 0 });
  svg.appendChild(cross);
  svg.setAttribute('aria-label', `Warm core at 300 hPa across ${n} analysis times for ${spec.series.map(s => s.name).join(', ')}`);
  hover(svg, tip, mx => {
    const i = Math.round((mx - L) / (W - L - R) * (n - 1));
    if (i < 0 || i >= n) return null;
    cross.setAttribute('x1', xs(i)); cross.setAttribute('x2', xs(i)); cross.setAttribute('opacity', 1);
    return [spec.labels[i], ...spec.series.map(s => `${s.name} ${isNum(s.values[i]) ? s.values[i].toFixed(2) + ' K' : 'n/a'}`), `ATMS ${spec.atms[i] ? 'yes' : 'none'}`];
  }, () => cross.setAttribute('opacity', 0));
}
