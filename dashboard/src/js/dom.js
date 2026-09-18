/** Small DOM helpers: build elements from data without innerHTML. */

/** h('div', {class:'tile'}, child, 'text', ...) */
export function h(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c != null) e.append(c.nodeType ? c : String(c));
  return e;
}

/** A results tile: number, unit, label. */
export function tile(n, unit, label) {
  return h('div', { class: 'tile' }, h('div', { class: 'n' }, n, unit ? h('span', { class: 'u' }, unit) : null), h('div', { class: 'l' }, label));
}

/** A table from a header row and body rows; cells may be strings or {text, class}. */
export function table(target, head, rows, caption) {
  const cell = (tag, c) => typeof c === 'object' && c !== null ? h(tag, { class: c.class }, c.text) : h(tag, {}, c);
  const t = target.tagName === 'TABLE' ? target : target.querySelector('table');
  t.replaceChildren(...[
    caption ? h('caption', { class: 'fine' }, caption) : null,
    h('thead', {}, h('tr', {}, ...head.map(c => cell('th', c)))),
    h('tbody', {}, ...rows.map(r => h('tr', {}, ...r.map(c => cell('td', c))))),
  ].filter(Boolean));
  return t;
}

export const fmt = (v, d = 2, fallback = 'n/a') => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : fallback);
