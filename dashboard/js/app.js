/**
 * Vaughan dashboard entry point.
 *
 * Load order: manifest.json (scalars, about 40 KB) first, so the tiles, tables and charts paint at once;
 * then the field buffer, so the scene explorer fills in; images stream in as the browser reaches them.
 * The imagery loop is driven by requestAnimationFrame and pauses when scrolled out of view or when the
 * user prefers reduced motion. Keyboard: the time strip is an arrow-key roving group; the play button
 * and slider are native controls.
 */
import { anomaly, getJSON, loadFields, rangeAbove, rowAnomaly, shifted } from './data.fdb4ce73.js';
import { mountFields } from './fields.42332423.js';
import { DIVERGING, GREY, SEQUENTIAL, gradient } from './colormap.4c095e89.js';
import { drawProfile, drawSeries } from './charts.1e0f3424.js';
import { fmt, h, table, tile } from './dom.5283f545.js';

const $ = id => document.getElementById(id);
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmtT = t => { const d = new Date(t + 'Z'); return `${String(d.getUTCDate()).padStart(2, '0')} ${MON[d.getUTCMonth()]} ${String(d.getUTCHours()).padStart(2, '0')}Z`; };
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const BIAS_CH7 = 18.29;   // ATMS channel 7 archive bias (K), see the audit table

async function main() {
  const status = $('status');
  const M = await getJSON('data/manifest.json');
  let S = M.scenes, storm = 'milton';                 // the storm shown in the explorer
  const PH = M.side.physics_only, V1 = M.side.v1_diverged, LEV = M.levels_hpa;
  const STORMS = M.storms || { milton: { label: 'Milton, October 2024', temperature_reference: 'ERA5', rain_reference: 'IMERG Final Run', scenes: M.scenes } };
  const tref = () => STORMS[storm].temperature_reference, rref = () => STORMS[storm].rain_reference;
  const i300 = LEV.indexOf(300);
  const covered = s => s.mw_coverage > 0.3;

  staticSections(M, M.scenes, PH, V1, LEV, i300);

  // ---- scene explorer state ----
  const views = mountFields(document);
  let cur = M.imagery.default_index;
  let fields = null;                                  // decoder for the storm on show, set once its buffer arrives
  const loaded = {};                                  // storm -> decoder, fetched on first use
  const X = { v: 'temp', level: 300, ref: 'era5', diff: false };    // variable, level, reference, difference

  const hero = { img: $('heroimg'), lab: $('herolab'), slider: $('heroslider'), idx: $('heroidx'), btn: $('playbtn') };
  const strip = $('strip');
  let chips = [];
  function buildStrip() {
    chips = S.map((s, i) => h('button', { class: 'chip', type: 'button', 'aria-pressed': 'false', tabindex: i === cur ? '0' : '-1', onclick: () => { setPlaying(false); select(i); } },
      h('span', { class: `dot ${covered(s) ? 'on' : ''}`, 'aria-hidden': 'true' }), fmtT(s.time), covered(s) ? h('span', { class: 'sr-only' }, ', ATMS overpass') : null));
    strip.replaceChildren(...chips);
  }
  buildStrip();
  // storm selector: Milton always; Melissa and Polo when their fields were packed by the build
  const ssel = $('stormsel');
  ssel.replaceChildren(...Object.keys(STORMS).map(k => h('option', { value: k }, STORMS[k].label)));
  ssel.addEventListener('change', e => { setPlaying(false); setStorm(e.target.value); });
  function setStorm(key) {
    if (!STORMS[key]) return;
    storm = key; S = STORMS[key].scenes;
    ssel.value = key;
    cur = key === 'milton' ? M.imagery.default_index : Math.max(0, S.findIndex(s => covered(s) && s.warm_core_K[i300] === Math.max(...S.filter(covered).map(x => x.warm_core_K[i300]))));
    buildStrip(); syncControls();
    fields = loaded[key] || null;
    select(cur);
    if (!fields) {
      status.hidden = false; status.textContent = `loading ${STORMS[key].label} fields`;
      loadFields(key === 'milton' ? M : STORMS[key]).then(d => { loaded[key] = d; if (storm === key) { fields = d; status.hidden = true; renderScene(cur); } })
        .catch(err => { status.textContent = `fields unavailable: ${err.message}`; });
    }
  }
  strip.addEventListener('keydown', e => {
    const k = { ArrowRight: 1, ArrowLeft: -1, Home: -Infinity, End: Infinity }[e.key];
    if (k === undefined) return;
    e.preventDefault(); setPlaying(false);
    select(k === -Infinity ? 0 : k === Infinity ? S.length - 1 : (cur + k + S.length) % S.length);
    chips[cur].focus();
  });
  $('prevbtn').addEventListener('click', () => { setPlaying(false); select((cur - 1 + S.length) % S.length); });
  $('nextbtn').addEventListener('click', () => { setPlaying(false); select((cur + 1) % S.length); });
  // variable tabs, level, reference, difference, phone switch
  const tabs = [...document.querySelectorAll('#vartabs .tab')];
  tabs.forEach(t => t.addEventListener('click', () => { X.v = t.dataset.var; tabs.forEach(u => u.setAttribute('aria-selected', String(u === t))); syncControls(); renderScene(cur); }));
  $('vartabs').addEventListener('keydown', e => {
    const k = { ArrowRight: 1, ArrowLeft: -1 }[e.key]; if (k === undefined) return;
    const i = tabs.findIndex(t => t.getAttribute('aria-selected') === 'true'); const n = tabs[(i + k + tabs.length) % tabs.length]; n.click(); n.focus(); e.preventDefault();
  });
  $('levelsel').addEventListener('change', e => { X.level = +e.target.value; syncControls(); renderScene(cur); });
  $('refsel').addEventListener('change', e => { X.ref = e.target.value; renderScene(cur); });
  $('diffchk').addEventListener('change', e => { X.diff = e.target.checked; renderScene(cur); });
  document.querySelectorAll('.phoneswitch .chip').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('.phoneswitch .chip').forEach(c => c.setAttribute('aria-pressed', String(c === b)));
    $('maps').dataset.show = b.dataset.show;
  }));
  function syncControls() {
    const temp = X.v === 'temp';
    $('levelgroup').hidden = !temp;
    $('refgroup').hidden = !temp;
    // physics-only and U-Net fields exist at 300 hPa only; the physics-only run was made for Milton only
    const phys = $('refsel').querySelector('[value=phys]'), un = $('refsel').querySelector('[value=unet]'), era = $('refsel').querySelector('[value=era5]');
    un.disabled = X.level !== 300;
    phys.disabled = X.level !== 300 || storm !== 'milton';
    era.textContent = tref();
    if ((X.level !== 300 && X.ref !== 'era5') || (X.ref === 'phys' && storm !== 'milton')) { X.ref = 'era5'; $('refsel').value = 'era5'; }
    $('diffchk').parentElement.hidden = !(X.v === 'temp' || X.v === 'rain');
    $('maps').dataset.var = X.v;
  }

  function showHero(i) {
    $('timelab').textContent = fmtT(S[i].time);
    if (storm !== 'milton') return;                 // the Gulf loop and the enhanced core image exist for Milton only
    const f = M.imagery.frames[i];
    hero.img.src = f.gulf;
    hero.img.alt = `GOES-16 band 13 enhanced infrared image of Hurricane Milton over the Gulf of Mexico, ${fmtT(f.time)} UTC`;
    hero.lab.textContent = `Milton · ${fmtT(f.time)} UTC`;
    hero.slider.value = i;
    hero.idx.textContent = `${i + 1} / ${S.length}`;
    $('coreimg').src = f.core;
  }

  function select(i) {
    cur = i;
    chips.forEach((c, j) => { c.setAttribute('aria-pressed', String(j === i)); c.tabIndex = j === i ? 0 : -1; });
    showHero(i);
    renderScene(i);
  }

  /** a minus b, same shape */
  const minus = (a, b) => { const data = new Float32Array(a.data.length); for (let k = 0; k < data.length; k++) data[k] = a.data[k] - b.data[k]; return { data, h: a.h, w: a.w }; };
  const REFNAME = () => ({ era5: tref(), unet: 'direct U-Net', phys: 'physics-only run' });
  function setScale(kind, lo, hi, units, note) {
    $('scale_grad').style.background = gradient(kind === 'div' ? DIVERGING : kind === 'seq' ? SEQUENTIAL : GREY);
    $('scale_lo').textContent = lo; $('scale_hi').textContent = hi; $('scale_units').textContent = units;
    if (note !== undefined) $('scale_note').textContent = note;
  }
  const wide = on => { for (const id of ['c_ana', 'c_ref']) views.get(id).canvas.classList.toggle('wide', on); };

  function renderScene(i) {
    const s = S[i];
    const t = fmtT(s.time);
    const e5 = s.warm_core_era5_K ? s.warm_core_era5_K[i300] : null;
    const st = [
      [isNum(s.inner_core_rmse_K.analysis) ? `${fmt(s.inner_core_rmse_K.analysis)} K` : 'no label', `inner-core RMSE at 300 hPa against ${tref()}, central 70 km (direct U-Net ${fmt(s.inner_core_rmse_K.unet)} K)`],
      [covered(s) ? `${Math.round(s.mw_coverage * 100)}%` : 'none', covered(s) ? 'of the domain seen by ATMS in the 90 minute window' : 'ATMS overpass in the window: infrared and prior only'],
      [`${fmt(s.warm_core_K[i300], 1)} K`, `warm core at 300 hPa, analysis; ${tref()} ${e5 == null ? 'n/a' : fmt(e5, 1) + ' K'}; member spread ${fmt(s.spread_core_300_K)} K`],
    ];
    $('scenestats').replaceChildren(...st.map(([n, l]) => h('div', { class: 'stat' }, h('div', { class: 'n' }, n), h('div', { class: 'l' }, l))));
    const d = isNum(s.inner_core_rmse_K.analysis) && isNum(s.inner_core_rmse_K.unet) ? s.inner_core_rmse_K.analysis - s.inner_core_rmse_K.unet : null;
    $('finding').textContent = covered(s)
      ? `At ${t} the sounder saw ${Math.round(s.mw_coverage * 100)} percent of the domain. The analysis puts the 300 hPa warm core at ${fmt(s.warm_core_K[i300], 1)} K against ${tref()}'s ${e5 == null ? 'unlabelled value' : fmt(e5, 1) + ' K'}${isNum(s.inner_core_rmse_K.analysis) ? `, with an inner-core error of ${fmt(s.inner_core_rmse_K.analysis)} K` : ''}${d == null ? '' : d < -0.02 ? `, ${fmt(-d)} K better than the direct U-Net` : d > 0.02 ? `, ${fmt(d)} K worse than the direct U-Net` : ', the same as the direct U-Net to 0.02 K'}.`
      : `At ${t} there was no ATMS overpass, so the analysis is the infrared through the learned proxy plus the prior. It puts the 300 hPa warm core at ${fmt(s.warm_core_K[i300], 1)} K against ${tref()}'s ${e5 == null ? 'unlabelled value' : fmt(e5, 1) + ' K'}${isNum(s.inner_core_rmse_K.analysis) ? `, inner-core error ${fmt(s.inner_core_rmse_K.analysis)} K` : ''}; the full system and the direct U-Net coincide here, as they must.`;
    drawProfile($('profile'), LEV, { era5: s.warm_core_era5_K, phys: storm === 'milton' ? PH[i]?.warm_core_K : null, unet: s.warm_core_unet_K, ana: s.warm_core_K });
    if (!fields) return;
    const has = k => fields.has(`s${i}.${k}`);
    const F = k => fields.get(`s${i}.${k}`);
    // the storm-centred infrared: Milton has the enhanced image, the other storms the band 13 field drawn in grey
    $('coreimg').hidden = storm !== 'milton'; views.get('c_ir').canvas.hidden = storm === 'milton';
    const A = views.get('c_ana'), R = views.get('c_ref');
    const ir = F('ir_obs_C13'); const irr = rangeAbove(ir, 50);
    $('irrange').textContent = irr ? `${irr[0].toFixed(0)} to ${irr[1].toFixed(0)} K` : '';
    if (storm !== 'milton' && irr) views.get('c_ir').draw(ir, 'grey', irr[0], irr[1], { invert: true, maskBelow: 50, label: `GOES band 13 observed brightness temperature, ${t}, cold cloud tops light` });
    wide(X.v === 'xsec');
    if (X.v === 'temp') {
      const lv = X.level, ana = anomaly(F(`T${lv}`));
      const refField = X.ref === 'era5' ? (has(`truth.T${lv}`) ? F(`truth.T${lv}`) : null) : X.ref === 'unet' ? F('T300_unet') : (storm === 'milton' && fields.has(`p${i}.T300`)) ? fields.get(`p${i}.T300`) : null;
      const ref = refField ? anomaly(refField) : null;
      $('lab_ana').textContent = `Analysis · ${lv} hPa`; $('sub_ana').textContent = 'temperature anomaly';
      A.draw(ana, 'div', -4, 4, { label: `Analysis ${lv} hPa temperature anomaly, ${t}, minus 4 to plus 4 K` });
      if (ref && X.diff) {
        $('lab_ref').textContent = `Analysis minus ${REFNAME()[X.ref]} · ${lv} hPa`; $('sub_ref').textContent = 'difference, own scale';
        R.draw(minus(ana, ref), 'div', -2, 2, { label: `Analysis minus ${REFNAME()[X.ref]} at ${lv} hPa, ${t}, minus 2 to plus 2 K` });
        setScale('div', '-2', '+2', 'K, difference', `left map on the anomaly scale (-4 to +4 K); right map: analysis minus ${REFNAME()[X.ref]}`);
      } else {
        $('lab_ref').textContent = `${REFNAME()[X.ref]} · ${lv} hPa`; $('sub_ref').textContent = ref ? 'temperature anomaly' : 'not available';
        if (ref) R.draw(ref, 'div', -4, 4, { label: `${REFNAME()[X.ref]} ${lv} hPa temperature anomaly, ${t}` });
        setScale('div', '-4', '+4', 'K from the domain mean', 'storm-centred box, about 560 km across; north up');
      }
    } else if (X.v === 'rain') {
      const ana = F('precip'), ref = has('truth.precip') ? F('truth.precip') : null;
      $('lab_ana').textContent = 'Analysis'; $('sub_ana').textContent = 'rain rate, ensemble mean';
      A.draw(ana, 'seq', 0, 30, { label: `Analysis rain rate, ${t}, 0 to 30 mm per hour` });
      if (ref && X.diff) {
        $('lab_ref').textContent = 'Analysis minus IMERG'; $('sub_ref').textContent = 'difference, own scale';
        R.draw(minus(ana, ref), 'div', -15, 15, { label: `Analysis minus IMERG rain rate, ${t}, minus 15 to plus 15 mm per hour` });
        setScale('div', '-15', '+15', 'mm/h, difference', 'left map 0 to 30 mm/h; right map: analysis minus IMERG');
      } else {
        $('lab_ref').textContent = 'IMERG'; $('sub_ref').textContent = ref ? rref() : 'not available';
        if (ref) R.draw(ref, 'seq', 0, 30, { label: `IMERG rain rate, ${t}` });
        setScale('seq', '0', '30', 'mm/h', 'storm-centred box, about 560 km across; north up');
      }
    } else if (X.v === 'obs') {
      const mo = F('mw_obs_7'), ms = F('mw_sim_7'); const mr = rangeAbove(mo, 50);
      $('lab_ana').textContent = 'ATMS channel 7 · observed'; $('sub_ana').textContent = '54.4 GHz, 16 x 16';
      $('lab_ref').textContent = 'ATMS channel 7 · simulated'; $('sub_ref').textContent = 'from the analysis, bias-corrected';
      if (mr) {
        A.draw(mo, 'div', mr[0], mr[1], { maskBelow: 50, label: `ATMS channel 7 observed, ${t}, ${mr[0].toFixed(0)} to ${mr[1].toFixed(0)} K` });
        const sim = shifted(ms, -BIAS_CH7); for (let k = 0; k < sim.data.length; k++) if (!(mo.data[k] > 50)) sim.data[k] = 0;   // show the simulation only inside the observed swath
        R.draw(sim, 'div', mr[0], mr[1], { maskBelow: 50, label: `ATMS channel 7 simulated from the analysis inside the observed swath, bias-corrected, ${t}` });
        setScale('div', mr[0].toFixed(0), mr[1].toFixed(0), 'K brightness temperature', 'scale set by the observed swath at this time; grey is outside the swath');
      } else {
        A.draw(mo, 'div', 0, 1, { maskBelow: 50, label: `ATMS channel 7: no overpass at ${t}` });
        R.draw(ms, 'div', 0, 1, { maskBelow: 50, label: `ATMS channel 7 simulated: no overpass at ${t}` });
        setScale('div', '', '', 'no ATMS overpass in the window', 'the infrared image on the left is the only observation at this time');
      }
    } else if (X.v === 'spread') {
      $('lab_ana').textContent = 'Ensemble spread · 300 hPa'; $('sub_ana').textContent = '8 members, standard deviation';
      A.draw(F('spread300'), 'seq', 0, 0.3, { label: `Ensemble spread at 300 hPa, ${t}, 0 to 0.3 K` });
      $('lab_ref').textContent = 'Analysis minus direct U-Net · 300 hPa'; $('sub_ref').textContent = 'what the prior and the physics changed';
      R.draw(minus(anomaly(F('T300')), anomaly(F('T300_unet'))), 'div', -1, 1, { label: `Analysis minus direct U-Net at 300 hPa, ${t}, minus 1 to plus 1 K` });
      setScale('seq', '0', '0.3', 'K spread (left)', 'right map: analysis minus direct U-Net, -1 to +1 K, diverging scale');
    } else if (X.v === 'xsec') {
      const ana = rowAnomaly(F('xsec_T')), ref = has('truth.xsec_T') ? rowAnomaly(F('truth.xsec_T')) : null;
      $('lab_ana').textContent = 'Analysis · east to west through the centre'; $('sub_ana').textContent = '200 to 1000 hPa';
      $('lab_ref').textContent = `${tref()} · east to west through the centre`; $('sub_ref').textContent = ref ? '200 to 1000 hPa' : 'not available';
      A.draw(ana, 'div', -4, 4, { label: `Analysis east-west temperature cross-section, anomaly from level mean, ${t}` });
      if (ref) R.draw(ref, 'div', -4, 4, { label: `${tref()} east-west temperature cross-section, ${t}` });
      setScale('div', '-4', '+4', 'K from each level mean', `top of each panel is 200 hPa, bottom 1000 hPa; the analysis has ten levels, ${tref()} ${ref ? ref.h : 'n/a'}`);
    }
  }

  // ---- imagery loop: requestAnimationFrame, paused off-screen and under reduced motion ----
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let playing = !reduced.matches, visible = true, last = 0, raf = 0;
  const PERIOD = 900;
  function frame(ts) {
    raf = 0;
    if (!playing || !visible) return;
    if (ts - last >= PERIOD) { last = ts; if (storm !== 'milton') setStorm('milton'); select((cur + 1) % S.length); }
    raf = requestAnimationFrame(frame);
  }
  function setPlaying(p) {
    playing = p;
    hero.btn.textContent = p ? 'Pause' : 'Play';
    hero.btn.setAttribute('aria-pressed', String(p));
    if (p && !raf) raf = requestAnimationFrame(frame);
  }
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible && playing && !raf) raf = requestAnimationFrame(frame); }, { threshold: 0.1 }).observe($('imagery'));
  reduced.addEventListener('change', e => setPlaying(!e.matches));
  hero.btn.addEventListener('click', () => setPlaying(!playing));
  hero.slider.addEventListener('input', e => { setPlaying(false); if (storm !== 'milton') setStorm('milton'); select(+e.target.value); });
  hero.slider.max = String(S.length - 1);

  syncControls();
  select(cur);
  setPlaying(playing);

  // ---- field buffer, then preload the remaining frames in the background ----
  status.hidden = false; status.textContent = 'loading fields';
  try {
    fields = await loadFields(M); loaded.milton = fields;
    status.hidden = true;
    renderScene(cur);
    priorSamples(M, loaded.milton, views);
  } catch (err) {
    status.textContent = `fields unavailable: ${err.message}`;
  }
  if ('requestIdleCallback' in window) requestIdleCallback(() => M.imagery.frames.forEach(f => { new Image().src = f.gulf; new Image().src = f.core; }));
}

/** Everything that needs only the manifest. */
function staticSections(M, S, PH, V1, LEV, i300) {
  const covered = s => s.mw_coverage > 0.3;
  // summary tiles
  const dom = S.map(s => s.domain_rmse_K.analysis).filter(isNum), core = S.map(s => s.inner_core_rmse_K.analysis).filter(isNum);
  const peak = S.map(s => LEV[s.warm_core_K.indexOf(Math.max(...s.warm_core_K))]);
  const at300 = peak.filter(p => p === 300 || p === 250 || p === 400).length;
  $('tiles').replaceChildren(
    tile(`${Math.min(...core).toFixed(1)} to ${Math.max(...core).toFixed(1)}`, 'K', `inner-core temperature RMSE at 300 hPa against ERA5, central 16 x 16 pixels (about 70 km), ${S.length} scenes; domain mean ${Math.min(...dom).toFixed(1)} to ${Math.max(...dom).toFixed(1)} K on the ${dom.length} with full labels`),
    tile(`${at300} of ${S.length}`, '', 'scenes with the warm core peaking at 250 to 400 hPa, where a mature hurricane keeps it'),
    tile(`${S.filter(covered).length} of ${S.length}`, '', 'analysis times with an ATMS overpass; the rest are infrared plus prior, with a 1.5 K sounding-channel misfit after calibration on the covered ones'),
  );
  // time series
  drawSeries($('series'), $('tip2'), {
    labels: S.map(s => fmtT(s.time).replace(/ [A-Z][a-z]{2} /, '/')), atms: S.map(covered),
    series: [
      { name: 'ERA5', values: S.map(s => s.warm_core_era5_K ? s.warm_core_era5_K[i300] : null), color: 'var(--s-era)', dashed: true },
      { name: 'first sampler', values: V1.map(s => s.warm_core_K[i300]), color: 'var(--s-v1)' },
      { name: 'physics-only', values: PH.map(s => s.warm_core_K[i300]), color: 'var(--s-phys)' },
      { name: 'U-Net', values: S.map(s => s.warm_core_unet_K[i300]), color: 'var(--s-unet)' },
      { name: 'analysis', values: S.map(s => s.warm_core_K[i300]), color: 'var(--s-ana)' },
    ],
    notes: [[6, 'peak intensity 07 Oct (897 hPa)'], [14, 'landfall 10 Oct 00:30Z']],
  });
  // error table
  table($('errtable'), ['time', 'ATMS', 'WC300 ana', 'WC300 U-Net', 'WC300 ERA5', 'core ana', 'core U-Net', 'core phys-only', 'domain ana', 'domain phys-only', 'domain first sampler', 'rain RMSE', 'spread 300'],
    S.map((s, i) => [fmtT(s.time), covered(s) ? `${Math.round(s.mw_coverage * 100)}%` : 'none', fmt(s.warm_core_K[i300], 1), fmt(s.warm_core_unet_K[i300], 1), s.warm_core_era5_K ? fmt(s.warm_core_era5_K[i300], 1) : 'n/a',
      fmt(s.inner_core_rmse_K.analysis), fmt(s.inner_core_rmse_K.unet), fmt(s.inner_core_rmse_K.physics_only), fmt(s.domain_rmse_K.analysis), fmt(PH[i].domain_rmse_K), fmt(V1[i].domain_rmse_K, 1), fmt(s.precip_rmse_vs_imerg, 1), fmt(s.spread_core_300_K)]));
  // comparison tiles
  const cov = S.map((s, i) => [s, i]).filter(([s]) => covered(s));
  const m = f => mean(cov.map(f).filter(isNum));
  const ana = m(([s]) => s.inner_core_rmse_K.analysis), un = m(([s]) => s.inner_core_rmse_K.unet), ph = m(([s]) => s.inner_core_rmse_K.physics_only);
  const wcA = m(([s]) => s.warm_core_K[i300]), wcP = m(([, i]) => PH[i].warm_core_K[i300]), wcE = m(([s]) => s.warm_core_era5_K ? s.warm_core_era5_K[i300] : null);
  const dA = m(([s]) => s.domain_rmse_K.analysis), dP = m(([, i]) => PH[i].domain_rmse_K), dV = m(([, i]) => V1[i].domain_rmse_K);
  $('cmp_tiles').replaceChildren(
    tile(ana.toFixed(2), 'K', `inner-core RMSE at 300 hPa, analysis, mean over the ${cov.length} scenes with ATMS`),
    tile(un.toFixed(2), 'K', 'same, direct U-Net proxy alone'),
    tile(ph.toFixed(2), 'K', 'same, physics-only microwave'),
    tile(`${wcA.toFixed(1)} / ${wcP.toFixed(1)} / ${wcE.toFixed(1)}`, 'K', 'warm core at 300 hPa: analysis / physics-only / ERA5'),
    tile(`${dA.toFixed(2)} / ${dP.toFixed(2)}`, 'K', 'domain RMSE, analysis / physics-only'),
    tile(dV.toFixed(1), 'K', 'domain RMSE of the first sampler before the guidance fix, same scenes'),
  );
  $('cmp_text').textContent = `On the ${cov.length} scenes with an ATMS overpass, the full system and the direct proxy are within ${Math.abs(ana - un).toFixed(2)} K of each other in the inner core (${ana.toFixed(2)} against ${un.toFixed(2)} K), and both beat the physics-only retrieval (${ph.toFixed(2)} K). Where there is no overpass all three coincide, as they must. The generative step does not improve on a microwave-aware proxy that is in distribution; it does turn an infrared-only proxy into a retrieval that fits the sounding channels to their noise floor, and it does so without retraining anything.`;
  // audit, probe and first-run tables
  const A = M.rtm_audit, sig = ch => (ch.startsWith('C') ? 5 : 2);
  const cls = (std, ch) => (std < 3 * sig(ch) ? 'ok' : std < 6 * sig(ch) ? 'warn' : 'bad'), lab = { ok: 'usable', warn: 'bias-correctable', bad: 'drop' };
  const name = { C08: 'ABI 8 · 6.2 µm', C10: 'ABI 10 · 7.3 µm', C13: 'ABI 13 · 10.3 µm', 5: 'ATMS 5 · 52.8 GHz', 6: 'ATMS 6 · 53.6', 7: 'ATMS 7 · 54.4', 8: 'ATMS 8 · 54.9', 9: 'ATMS 9 · 55.5', 16: 'ATMS 16 · 88', 17: 'ATMS 17 · 165', 18: 'ATMS 18 · 183±7', 22: 'ATMS 22 · 183±1' };
  table($('audit'), ['channel', 'bias 2023', 'scatter 2023', 'bias Milton', 'scatter Milton', 'verdict'],
    Object.keys(A.archive_2023).map(ch => { const a = A.archive_2023[ch], mm = A.milton[ch], c = cls(a[2], ch); return [name[ch] ?? ch, fmt(a[0], 1), fmt(a[2], 1), fmt(mm[0], 1), fmt(mm[2], 1), { text: lab[c], class: c }]; }));
  const P = M.probe;
  table($('probe'), ['single chain, 08 Oct 18Z', 'T RMSE vs ERA5', 'ATMS misfit', '|x| at end'], [
    ['original guidance', { text: `${P.before.T_rmse_K} K`, class: 'bad' }, { text: `${P.before.mw_misfit_K} K`, class: 'bad' }, `${P.before.x_rms_final} sigma`],
    ['Tweedie-inflated, gated', { text: `${P.after.T_rmse_K} K`, class: 'ok' }, { text: `${P.after.mw_misfit_K} K`, class: 'ok' }, `${P.after.x_rms_final} sigma`]]);
  table($('v1'), ['scene', 'misfit U-Net', 'misfit analysis', 'misfit ERA5', 'T err U-Net', 'T err analysis'],
    M.v1.rows.map(r => [fmtT(`${r[0]}:00`), r[1], { text: r[2], class: 'bad' }, r[3], r[4], { text: r[5], class: 'bad' }]));
}

function priorSamples(M, fields, views) {
  const g = $('priorgrid');
  const n = M.prior_samples.n;
  g.replaceChildren(...Array.from({ length: n }, (_, j) => h('div', { class: 'panel' },
    h('div', { class: 't' }, h('span', {}, `draw ${j + 1}`), h('span', {}, 'no observations')),
    h('div', { class: 'pair' },
      h('figure', {}, h('canvas', { class: 'field', id: `pr_t${j}` }), h('figcaption', {}, 'T 400 hPa anomaly, K')),
      h('figure', {}, h('canvas', { class: 'field', id: `pr_p${j}` }), h('figcaption', {}, 'rain, mm/h'))))));
  const pv = mountFields(g);
  for (let j = 0; j < n; j++) {
    pv.get(`pr_t${j}`).draw(anomaly(fields.get(`prior.T400.${j}`)), 'div', -4, 4, { label: `Prior draw ${j + 1}: 400 hPa temperature anomaly` });
    pv.get(`pr_p${j}`).draw(fields.get(`prior.precip.${j}`), 'seq', 0, 30, { label: `Prior draw ${j + 1}: rain rate` });
  }
  pv.forEach((v, k) => views.set(k, v));
}

main().catch(err => { const s = $('status'); s.hidden = false; s.textContent = `The dashboard could not load its data: ${err.message}`; console.error(err); });

if ('serviceWorker' in navigator && location.protocol === 'https:') {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
