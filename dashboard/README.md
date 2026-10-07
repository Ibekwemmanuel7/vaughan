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

The page is organised as four views (Overview, Explorer, Results, Methods) with the long tables behind
`<details>` disclosures. The explorer shows one variable at a time as two large maps, the analysis on the
left and its reference on the right on one colour scale (Temperature at 300 or 850 hPa against ERA5, the
direct U-Net or the physics-only run; Rain against IMERG; Observations as ATMS channel 7 observed and
simulated inside the swath; Spread beside the analysis-minus-U-Net difference; the east-west cross-section),
with a difference toggle on its own scale, three per-scene metrics and a one-sentence finding. On narrow
screens one map is shown with an Analysis/Reference switch. Milton only, since only its fields are packed.


Polo (September 2026) is a static section like Melissa's: figures from results/polo (polo_summary.png, polo_structure_2026-09-22T2000.png copied to raw/polo_structure.png) and the per-scene table generated from results/polo/polo_scores.csv. The time-continuity subsection uses results/polo/polo_persist.png (copied to raw/) and the numbers from polo_persist_compare.csv.
