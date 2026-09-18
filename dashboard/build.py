"""
Build the Vaughan dashboard.

  python build.py            -> out/

  python build.py --in-place -> this folder (what GitHub Pages serves)

Inputs: src/ (shell, stylesheet, section markup, modules), raw/dash.json (scores and fields),
raw/imagery.json (base64 JPEG frames), raw/*.png (the two figures) and logo/.

Outputs (all static, no framework):
  index.html                 shell with sections, content-hashed asset links
  styles.<hash>.css, js/*.js
  data/manifest.json         scalars, tables, field index
  data/fields.<hash>.bin     every 2-D field as Uint16 quantised on [lo, hi], concatenated
  img/gulf/NN.webp, img/core/NN.webp, img/*.webp, img/og.png, img/icon-*.png
  sw.js, manifest.webmanifest, favicon.svg
"""
import base64, hashlib, io, json, os, re, shutil, struct, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
OUT = os.path.join(HERE, 'out') if '--in-place' not in sys.argv else HERE
RAW = os.path.join(HERE, 'raw')
DASH = os.path.join(RAW, 'dash.json')
IMAGERY = os.path.join(RAW, 'imagery.json')
FIGS = {'melissa_summary': os.path.join(RAW, 'melissa_summary.png'), 'weathernext_milton': os.path.join(RAW, 'weathernext_milton.png')}
LOGO_PNG = os.path.join(HERE, 'logo', 'vaughan_mark.png')
LOGO_ON_NAVY = os.path.join(HERE, 'logo', 'vaughan_mark_on_navy.png')

SCENE_FIELDS = ['T300', 'T850', 'precip', 'xsec_T', 'ir_obs_C13', 'mw_obs_7', 'mw_sim_7', 'T300_unet', 'spread300']
TRUTH_FIELDS = ['T300', 'T850', 'precip', 'xsec_T']


def h8(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]


class FieldPacker:
    """Concatenate 2-D arrays as Uint16 on [lo, hi] with per-array metadata."""

    def __init__(self):
        self.chunks, self.index, self.off = [], {}, 0

    def add(self, key, arr):
        a = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(a)
        lo = float(a[finite].min()) if finite.any() else 0.0
        hi = float(a[finite].max()) if finite.any() else 1.0
        if hi <= lo:
            hi = lo + 1.0
        q = np.zeros(a.shape, dtype=np.uint16)
        q[finite] = np.round((a[finite] - lo) / (hi - lo) * 65535).astype(np.uint16)
        # masked pixels are already encoded as values below 50 K in the data; NaN never occurs
        assert finite.all(), f'{key}: non-finite values'
        b = q.astype('<u2').tobytes()
        self.index[key] = {'lo': lo, 'hi': hi, 'off': self.off // 2, 'shape': list(a.shape)}  # offset in Uint16 elements
        self.chunks.append(b)
        self.off += len(b)

    def bytes(self):
        return b''.join(self.chunks)


def main():
    if OUT != HERE and os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for d in ('js', 'data', 'img/gulf', 'img/core'):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    if OUT == HERE:  # in place: drop previously hashed assets so only the current build remains
        for pat in ('styles.*.css', 'js/*.*.js', 'data/fields.*.bin'):
            import glob
            for p in glob.glob(os.path.join(OUT, pat)):
                os.remove(p)

    D = json.load(open(DASH))
    S = D['runs']['milton_v3']
    PH = D['side']['physics_only']

    # ---- fields ----
    pk = FieldPacker()
    for i, s in enumerate(S):
        for k in SCENE_FIELDS:
            pk.add(f's{i}.{k}', s['fields'][k])
        for k in TRUTH_FIELDS:
            pk.add(f's{i}.truth.{k}', s['truth'][k])
        if PH[i].get('fields', {}).get('T300') is not None:
            pk.add(f'p{i}.T300', PH[i]['fields']['T300'])
    P = D['prior_samples']
    for j in range(len(P['T400'])):
        pk.add(f'prior.T400.{j}', P['T400'][j])
        pk.add(f'prior.precip.{j}', P['precip'][j])
    fb = pk.bytes()
    fields_name = f'fields.{h8(fb)}.bin'
    open(os.path.join(OUT, 'data', fields_name), 'wb').write(fb)

    # ---- imagery -> webp ----
    im = json.load(open(IMAGERY))
    frames = []
    for i, f in enumerate(im['frames']):
        entry = {'time': f['time']}
        for k, sub in (('gulf', 'gulf'), ('core', 'core')):
            raw = base64.b64decode(f[k].split(',', 1)[1])
            img = Image.open(io.BytesIO(raw)).convert('RGB')
            rel = f'img/{sub}/{i:02d}.webp'
            img.save(os.path.join(OUT, rel), 'WEBP', quality=82, method=6)
            entry[k] = rel
        frames.append(entry)
    for name, path in FIGS.items():
        img = Image.open(path).convert('RGB')
        img.save(os.path.join(OUT, f'img/{name}.webp'), 'WEBP', quality=88, method=6)

    # ---- Open Graph image and PWA icons ----
    og = Image.new('RGB', (1200, 630), (30, 47, 74))
    hero = Image.open(io.BytesIO(base64.b64decode(im['frames'][im.get('default_index', 10)]['gulf'].split(',', 1)[1]))).convert('RGB')
    hero = hero.resize((int(630 * hero.width / hero.height), 630))
    hero = hero.crop((hero.width - 620, 0, hero.width, 630))
    og.paste(hero, (1200 - hero.width, 0))
    mark = Image.open(LOGO_ON_NAVY).convert('RGBA')
    mark = mark.resize((180, int(180 * mark.height / mark.width)))
    og.paste(mark, (60, 50), mark)
    dr = ImageDraw.Draw(og)
    from PIL import ImageFont
    def font(sz):
        for p in ('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'):
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()
    dr.text((60, 250), 'VAUGHAN', font=font(72), fill=(242, 244, 247))
    dr.text((60, 350), 'Milton Inside Out', font=font(40), fill=(242, 244, 247))
    dr.text((60, 410), 'a satellite-only hurricane structure engine', font=font(28), fill=(200, 208, 218))
    og.save(os.path.join(OUT, 'img/og.png'), optimize=True)
    for sz in (180, 192, 512):
        ic = Image.new('RGBA', (sz, sz), (242, 244, 247, 255))
        m = Image.open(LOGO_PNG).convert('RGBA')
        w = int(sz * 0.8); m = m.resize((w, int(w * m.height / m.width)))
        ic.paste(m, ((sz - m.width) // 2, (sz - m.height) // 2), m)
        ic.save(os.path.join(OUT, f'img/icon-{sz}.png'), optimize=True)

    # ---- manifest: everything except the fields ----
    def slim_scene(s):
        return {k: v for k, v in s.items() if k not in ('fields', 'truth')}
    M = {
        'version': 1,
        'levels_hpa': S[0]['levels_hpa'],
        'scenes': [slim_scene(s) for s in S],
        'side': {'physics_only': [{k: v for k, v in p.items() if k != 'fields'} for p in PH], 'v1_diverged': D['side']['v1_diverged']},
        'rtm_audit': D['rtm_audit'], 'v1': D['v1'], 'probe': D['probe'], 'training': D['training'],
        'prior_samples': {'n': len(P['T400']), 'levels_hpa': P['levels_hpa']},
        'imagery': {'frames': frames, 'gulf_extent': im['gulf_extent'], 'default_index': 10},
        'fields': {'file': f'data/{fields_name}', 'bytes': len(fb), 'dtype': 'uint16-le', 'index': pk.index},
    }
    mb = json.dumps(M, separators=(',', ':')).encode()
    open(os.path.join(OUT, 'data', 'manifest.json'), 'wb').write(mb)

    # ---- static assets with content hashes ----
    css = open(os.path.join(SRC, 'styles.css'), 'rb').read()
    css_name = f'styles.{h8(css)}.css'
    open(os.path.join(OUT, css_name), 'wb').write(css)
    js_names = {}
    for fn in sorted(os.listdir(os.path.join(SRC, 'js'))):
        b = open(os.path.join(SRC, 'js', fn), 'rb').read()
        js_names[fn] = b
    # rewrite relative imports to hashed names in two passes (hash after rewriting dependencies)
    order = ['colormap.js', 'dom.js', 'data.js', 'fields.js', 'charts.js', 'app.js']
    hashed = {}
    for fn in order:
        src = js_names[fn].decode()
        for dep, hname in hashed.items():
            src = src.replace(f"'./{dep}'", f"'./{hname}'")
        b = src.encode()
        hname = fn if fn == 'app.js' else f'{fn[:-3]}.{h8(b)}.js'
        hashed[fn] = hname
        open(os.path.join(OUT, 'js', hname), 'wb').write(b)
    app_hash = h8(open(os.path.join(OUT, 'js', 'app.js'), 'rb').read())
    for fn in ('favicon.svg', 'manifest.webmanifest'):
        shutil.copy(os.path.join(SRC, fn), os.path.join(OUT, fn))

    # ---- shell ----
    html = open(os.path.join(SRC, 'index.html')).read()
    html = html.replace('<!--SECTIONS-->', open(os.path.join(SRC, 'sections.html')).read().strip())
    html = html.replace('<!--FOOTER-->', open(os.path.join(SRC, 'footer.html')).read().strip())
    html = html.replace('href="styles.css"', f'href="{css_name}"')
    html = html.replace('src="js/app.js"', f'src="js/app.js?v={app_hash}"').replace('href="js/app.js"', f'href="js/app.js?v={app_hash}"')
    assert '<!--' not in html.replace('<!--SECTIONS-->', ''), 'unfilled placeholder'
    open(os.path.join(OUT, 'index.html'), 'w').write(html)

    # ---- service worker ----
    shell = ['./', 'index.html', css_name, 'favicon.svg', 'data/manifest.json', f'data/{fields_name}'] + [f'js/{n}' for n in hashed.values()]
    shell += [fr['gulf'] for fr in frames] + [fr['core'] for fr in frames] + [f'img/{n}.webp' for n in FIGS]
    version = h8(b''.join(open(os.path.join(OUT, p), 'rb').read() for p in ['index.html', css_name, 'data/manifest.json'] + [f'js/{n}' for n in hashed.values()]))
    sw = open(os.path.join(SRC, 'sw.js')).read().replace('__VERSION__', version).replace('__SHELL__', json.dumps(shell))
    open(os.path.join(OUT, 'sw.js'), 'w').write(sw)

    # ---- report ----
    total = 0
    rows = []
    for root, _, files in os.walk(OUT):
        rel = os.path.relpath(root, OUT)
        if rel.split(os.sep)[0] in ('src', 'raw', 'tests', 'logo', 'out', '__pycache__', '.pytest_cache'):
            continue
        for f in files:
            if root == OUT and f in ('build.py', 'README.md'):
                continue
            p = os.path.join(root, f); n = os.path.getsize(p); total += n
            rows.append((os.path.relpath(p, OUT), n))
    big = sorted(rows, key=lambda r: -r[1])[:8]
    print(f'out: {len(rows)} files, {total/1e6:.2f} MB total')
    print(f'  index.html {os.path.getsize(os.path.join(OUT, "index.html"))/1e3:.0f} KB, css {len(css)/1e3:.0f} KB, manifest {len(mb)/1e3:.0f} KB, fields {len(fb)/1e6:.2f} MB')
    gulf = sum(n for p, n in rows if p.startswith('img/gulf')); core = sum(n for p, n in rows if p.startswith('img/core'))
    print(f'  gulf frames {gulf/1e3:.0f} KB, core frames {core/1e3:.0f} KB (16 each)')
    for p, n in big:
        print(f'  {n/1e3:8.0f} KB  {p}')


if __name__ == '__main__':
    main()
