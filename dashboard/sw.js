/* Vaughan dashboard service worker: shell and data cache-first with a versioned cache, the page itself network-first. */
const VERSION = 'fe0b3681';
const CACHE = `vaughan-${VERSION}`;
const SHELL = ["./", "index.html", "styles.0120ec49.css", "favicon.svg", "data/manifest.json", "data/fields.9cea091c.bin", "js/colormap.4c095e89.js", "js/dom.5283f545.js", "js/data.fdb4ce73.js", "js/fields.42332423.js", "js/charts.1e0f3424.js", "js/app.js", "img/gulf/00.webp", "img/gulf/01.webp", "img/gulf/02.webp", "img/gulf/03.webp", "img/gulf/04.webp", "img/gulf/05.webp", "img/gulf/06.webp", "img/gulf/07.webp", "img/gulf/08.webp", "img/gulf/09.webp", "img/gulf/10.webp", "img/gulf/11.webp", "img/gulf/12.webp", "img/gulf/13.webp", "img/gulf/14.webp", "img/gulf/15.webp", "img/core/00.webp", "img/core/01.webp", "img/core/02.webp", "img/core/03.webp", "img/core/04.webp", "img/core/05.webp", "img/core/06.webp", "img/core/07.webp", "img/core/08.webp", "img/core/09.webp", "img/core/10.webp", "img/core/11.webp", "img/core/12.webp", "img/core/13.webp", "img/core/14.webp", "img/core/15.webp", "img/melissa_summary.webp", "img/weathernext_milton.webp", "img/polo_summary.webp", "img/polo_structure.webp"];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin) return;
  const isPage = e.request.mode === 'navigate' || url.pathname.endsWith('/') || url.pathname.endsWith('index.html');
  if (isPage) {
    e.respondWith(fetch(e.request).then(r => { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return r; }).catch(() => caches.match(e.request)));
    return;
  }
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(r => {
    if (r.ok) { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
    return r;
  })));
});
