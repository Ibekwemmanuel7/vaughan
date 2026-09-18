# Vaughan dashboard

The public page for the Vaughan platform: `https://ibekwemmanuel7.github.io/vaughan/dashboard/`.

A static site with no framework and no build-time dependencies beyond Python, NumPy, Pillow
and, for the tests, Playwright.

## Layout

    index.html              the page (shell + sections, asset links content-hashed)
    styles.<hash>.css
    js/app.js               entry (ES module); the other modules are content-hashed
    data/manifest.json      scalars, tables, field index (about 35 KB)
    data/fields.<hash>.bin  every 2-D field, Uint16 quantised per array (about 1.4 MB)
    img/gulf/NN.webp        16 GOES-16 Gulf frames; img/core/NN.webp storm-centred
    img/*.webp, img/og.png  figures and the Open Graph card
    sw.js                   service worker (cache-first assets, network-first page)
    manifest.webmanifest, favicon.svg
    src/                    what you edit: index.html shell, sections.html, footer.html,
                            styles.css, js/*.js, sw.js
    raw/                    build inputs: dash.json, imagery.json, the two figure PNGs
    logo/                   the Vaughan mark
    build.py, tests/

## Build

    python build.py              # writes out/ for a look
    python build.py --in-place   # writes the site into this folder (what Pages serves)
    python -m pytest tests -q    # Playwright smoke test against out/

## What the page does

The shell paints from `manifest.json` (tiles, tables, charts), then the field buffer arrives and
the scene explorer fills in; images stream in lazily. Fields are drawn on offscreen canvases at
native resolution and blitted at device-pixel size (ResizeObserver, DPR-aware). The imagery loop
runs on requestAnimationFrame, pauses off-screen (IntersectionObserver) and under
prefers-reduced-motion. Colour maps are Moreland cool-warm and viridis. Everything interactive
is a native control or an ARIA-labelled canvas; the time strip is an arrow-key roving group.
Strict CSP (no inline script or style), dark scheme, print rules.
