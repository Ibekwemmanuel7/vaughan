"""
Smoke test for the built dashboard (out/, or this folder when built in place). Serves it over HTTP and drives it with Playwright.

  python -m pytest tests -q
"""
import http.server, os, socketserver, threading, functools
import pytest
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out') if os.path.isdir(os.path.join(ROOT, 'out')) else ROOT  # out/ from build.py, else the in-place build


@pytest.fixture(scope='session')
def server():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
    handler = functools.partial(Quiet, directory=OUT)
    with socketserver.TCPServer(('127.0.0.1', 0), handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
        yield f'http://127.0.0.1:{port}/'
        httpd.shutdown()


@pytest.fixture(scope='session')
def pw():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


def load(pw, server, width=1280, height=900, dpr=1):
    ctx = pw.new_context(viewport={'width': width, 'height': height}, device_scale_factor=dpr)
    page = ctx.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    page.goto(server, wait_until='networkidle')
    page.wait_for_function("document.getElementById('status').hidden === true", timeout=15000)
    return page, errors


def test_loads_without_errors(pw, server):
    page, errors = load(pw, server)
    assert errors == [], errors
    assert page.locator('#tiles .tile').count() == 6
    assert page.locator('#errtable tbody tr').count() == 16
    assert page.locator('#audit tbody tr').count() == 12
    assert page.locator('#strip .chip').count() == 16
    assert page.locator('#priorgrid canvas').count() == 8
    assert 'null' not in page.locator('#errtable').inner_text()


def test_canvases_are_labelled_and_painted(pw, server):
    page, _ = load(pw, server, dpr=2)
    labels = page.eval_on_selector_all('canvas.field', 'cs => cs.map(c => [c.getAttribute("role"), c.getAttribute("aria-label"), c.width, c.clientWidth])')
    assert labels, 'no canvases'
    for role, label, w, cw in labels:
        assert role == 'img' and label and len(label) > 10, (role, label)
        assert w >= 2 * cw - 2, f'canvas not DPR-aware: backing {w} for css {cw}'
    # a painted canvas is not uniformly the background colour
    nonuniform = page.eval_on_selector('#c_t3a', '''c => { const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data; const s = new Set(); for (let i=0;i<d.length;i+=4*97) s.add(d[i]<<16|d[i+1]<<8|d[i+2]); return s.size; }''')
    assert nonuniform > 10


def test_controls(pw, server):
    page, errors = load(pw, server)
    page.click('#playbtn')
    assert page.locator('#playbtn').get_attribute('aria-pressed') == 'false'
    page.locator('#heroslider').fill('3')
    page.locator('#heroslider').dispatch_event('input')
    assert page.locator('#heroidx').inner_text() == '4 / 16'
    assert page.locator('#strip .chip').nth(3).get_attribute('aria-pressed') == 'true'
    assert '07 Oct' in page.locator('#herolab').inner_text()
    page.locator('#strip .chip').nth(3).focus()
    page.keyboard.press('ArrowRight')
    assert page.locator('#strip .chip').nth(4).get_attribute('aria-pressed') == 'true'
    assert page.evaluate('document.activeElement.textContent') == page.locator('#strip .chip').nth(4).inner_text()
    assert errors == []


def test_no_horizontal_overflow_on_phone(pw, server):
    page, errors = load(pw, server, width=390, height=844, dpr=3)
    sw, cw = page.evaluate('[document.documentElement.scrollWidth, document.documentElement.clientWidth]')
    assert sw <= cw, f'page overflows: {sw} > {cw}'
    wide = page.evaluate('''() => [...document.querySelectorAll('.tbl')].map(t => [t.scrollWidth > t.clientWidth, t.getBoundingClientRect().right <= innerWidth])''')
    assert all(fits for _, fits in wide), wide
    assert any(scrolls for scrolls, _ in wide), 'expected at least one table to scroll inside its wrapper'
    assert errors == []


def test_csp_and_meta(pw, server):
    page, _ = load(pw, server)
    csp = page.get_attribute('meta[http-equiv="Content-Security-Policy"]', 'content')
    assert "script-src 'self'" in csp and 'unsafe-inline' not in csp
    assert page.get_attribute('meta[property="og:image"]', 'content')
    assert page.locator('main').count() == 1 and page.locator('a.skip').count() == 1
    assert page.evaluate("document.fonts.check('16px \"Times New Roman\"') || true")


def test_reduced_motion_pauses_loop(pw, server):
    ctx = pw.new_context(viewport={'width': 1280, 'height': 900}, reduced_motion='reduce')
    page = ctx.new_page(); page.goto(server, wait_until='networkidle')
    assert page.locator('#playbtn').inner_text() == 'Play'


def test_polo_section(pw, server):
    page, errors = load(pw, server)
    assert page.locator('#polo table tbody tr').count() == 31
    assert page.locator('#polo tbody tr td:nth-child(2)', has_text='yes').count() == 9
    page.locator('#polo').scroll_into_view_if_needed()      # the figures are lazy-loaded
    page.locator('#polo .fine').last.scroll_into_view_if_needed()
    page.wait_for_timeout(800)
    for name in ('polo_summary', 'polo_structure'):
        ok = page.evaluate(f"() => {{ const i = document.querySelector('img[src=\"img/{name}.webp\"]'); return i && i.complete && i.naturalWidth > 0 }}")
        assert ok, name
    assert page.locator('nav a[href="#polo"]').count() >= 1
    assert errors == [], errors
