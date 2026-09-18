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
import { drawProfile, drawSeries } from './charts.1e0f3424.js';
import { fmt, h, table, tile } from './dom.5283f545.js';

const $ = id => document.getElementById(id);
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const fmtT = t => { const d = new Date(t + 'Z'); return `${String(d.getUTCDate()).padStart(2, '0')} Oct ${String(d.getUTCHours()).padStart(2, '0')}Z`; };
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const BIAS_CH7 = 18.29;   // ATMS channel 7 archive bias (K), see the audit table

async function main() {
  const status = $('status');
  const M = await getJSON('data/manifest.json');
  const S = M.scenes, PH = M.side.physics_only, V1 = M.side.v1_diverged, LEV = M.levels_hpa;
  const i300 = LEV.indexOf(300);
  const covered = s => s.mw_coverage > 0.3;

  staticSections(M, S, PH, V1, LEV, i300);

  // ---- scene explorer state ----
  const views = mountFields(document);
  let cur = M.imagery.default_index;
  let fields = null;                                  // set once the buffer arrives

  const hero = { img: $('heroimg'), lab: $('herolab'), slider: $('heroslider'), idx: $('heroidx'), btn: $('playbtn') };
  const strip = $('strip');
  const chips = S.map((s, i) => h('button', { class: 'chip', type: 'button', 'aria-pressed': 'false', tabindex: i === cur ? '0' : '-1', onclick: () => { setPlaying(false); select(i); } },
    h('span', { class: `dot ${covered(s) ? 'on' : ''}`, 'aria-hidden': 'true' }), fmtT(s.time), covered(s) ? h('span', { class: 'sr-only' }, ', ATMS overpass') : null));
  chips.forEach(c => strip.append(c));
  strip.addEventListener('keydown', e => {
    const k = { ArrowRight: 1, ArrowLeft: -1, Home: -Infinity, End: Infinity }[e.key];
    if (k === undefined) return;
    e.preventDefault(); setPlaying(false);
    select(k === -Infinity ? 0 : k === Infinity ? S.length - 1 : (cur + k + S.length) % S.length);
    chips[cur].focus();
  });

  function showHero(i) {
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

  function renderScene(i) {
    const s = S[i];
    const st = [
      [fmtT(s.time), 'analysis time (UTC)'],
      [covered(s) ? `${Math.round(s.mw_coverage * 100)}%` : 'none', 'ATMS coverage of the domain'],
      [isNum(s.inner_core_rmse_K.analysis) ? `${fmt(s.inner_core_rmse_K.analysis)} K` : 'no label', `inner-core RMSE at 300 hPa (physics-only ${fmt(s.inner_core_rmse_K.physics_only)} K)`],
      [isNum(s.domain_rmse_K.analysis) ? `${fmt(s.domain_rmse_K.analysis)} K` : 'no label', 'domain RMSE vs ERA5'],
      [isNum(s.precip_rmse_vs_imerg) ? `${fmt(s.precip_rmse_vs_imerg, 1)} mm/h` : 'no label', 'rain RMSE vs IMERG'],
      [`${fmt(s.warm_core_K[i300], 1)} K`, `warm core at 300 hPa (ERA5 ${s.warm_core_era5_K ? fmt(s.warm_core_era5_K[i300], 1) : 'n/a'} K)`],
    ];
    $('scenestats').replaceChildren(...st.map(([n, l]) => h('div', { class: 'stat' }, h('div', { class: 'n' }, n), h('div', { class: 'l' }, l))));
    drawProfile($('profile'), LEV, { era5: s.warm_core_era5_K, phys: PH[i]?.warm_core_K, unet: s.warm_core_unet_K, ana: s.warm_core_K });
    if (!fields) return;
    const t = fmtT(s.time);
    const F = k => fields.get(`s${i}.${k}`);
    const ir = F('ir_obs_C13'); const irr = rangeAbove(ir, 50);
    $('irrange').textContent = irr ? `${irr[0].toFixed(0)} to ${irr[1].toFixed(0)} K` : '';
    const mo = F('mw_obs_7'), ms = F('mw_sim_7'); const mr = rangeAbove(mo, 50);
    if (mr) {
      views.get('c_mwo').draw(mo, 'div', mr[0], mr[1], { maskBelow: 50, label: `ATMS channel 7 observed, ${t}, ${mr[0].toFixed(0)} to ${mr[1].toFixed(0)} K` });
      views.get('c_mws').draw(shifted(ms, -BIAS_CH7), 'div', mr[0], mr[1], { maskBelow: 50, label: `ATMS channel 7 simulated from the analysis, bias-corrected, ${t}` });
    } else {
      views.get('c_mwo').draw(mo, 'div', 0, 1, { maskBelow: 50, label: `ATMS channel 7: no overpass at ${t}` });
      views.get('c_mws').draw(ms, 'div', 0, 1, { maskBelow: 50, label: `ATMS channel 7 simulated: no overpass at ${t}` });
    }
    views.get('c_t3a').draw(anomaly(F('T300')), 'div', -4, 4, { label: `Analysis 300 hPa temperature anomaly, ${t}, minus 4 to plus 4 K` });
    views.get('c_t3e').draw(anomaly(F('truth.T300')), 'div', -4, 4, { label: `ERA5 300 hPa temperature anomaly, ${t}` });
    views.get('c_t8a').draw(anomaly(F('T850')), 'div', -4, 4, { label: `Analysis 850 hPa temperature anomaly, ${t}` });
    views.get('c_t8e').draw(anomaly(F('truth.T850')), 'div', -4, 4, { label: `ERA5 850 hPa temperature anomaly, ${t}` });
    views.get('c_pa').draw(F('precip'), 'seq', 0, 30, { label: `Analysis rain rate, ${t}, 0 to 30 mm per hour` });
    views.get('c_pe').draw(F('truth.precip'), 'seq', 0, 30, { label: `IMERG rain rate, ${t}` });
    views.get('c_xa').draw(rowAnomaly(F('xsec_T')), 'div', -4, 4, { label: `Analysis east-west temperature cross-section, anomaly from level mean, ${t}` });
    views.get('c_xe').draw(rowAnomaly(F('truth.xsec_T')), 'div', -4, 4, { label: `ERA5 east-west temperature cross-section, ${t}` });
    views.get('c_sp').draw(F('spread300'), 'seq', 0, 0.5, { label: `Ensemble spread at 300 hPa, ${t}, 0 to 0.5 K` });
    if (fields.has(`p${i}.T300`)) views.get('c_t3p').draw(anomaly(fields.get(`p${i}.T300`)), 'div', -4, 4, { label: `Physics-only run, 300 hPa anomaly, ${t}` });
  }

  // ---- imagery loop: requestAnimationFrame, paused off-screen and under reduced motion ----
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let playing = !reduced.matches, visible = true, last = 0, raf = 0;
  const PERIOD = 900;
  function frame(ts) {
    raf = 0;
    if (!playing || !visible) return;
    if (ts - last >= PERIOD) { last = ts; select((cur + 1) % S.length); }
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
  hero.slider.addEventListener('input', e => { setPlaying(false); select(+e.target.value); });
  hero.slider.max = String(S.length - 1);

  select(cur);
  setPlaying(playing);

  // ---- field buffer, then preload the remaining frames in the background ----
  status.hidden = false; status.textContent = 'loading fields';
  try {
    fields = await loadFields(M);
    status.hidden = true;
    renderScene(cur);
    priorSamples(M, fields, views);
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
    tile(`${Math.min(...dom).toFixed(1)} to ${Math.max(...dom).toFixed(1)}`, 'K', 'domain temperature RMSE against ERA5, 14 scored scenes'),
    tile(`${Math.min(...core).toFixed(1)} to ${Math.max(...core).toFixed(1)}`, 'K', 'inner-core temperature RMSE at 300 hPa, central 16 x 16 pixels (about 70 km)'),
    tile(`${at300} of ${S.length}`, '', 'scenes with the warm core peaking at 250 to 400 hPa, where it belongs'),
    tile('1.5', 'K', 'ATMS sounding-channel misfit after calibration; the ERA5 truth itself sits at 2.5 to 3 K'),
    tile(`${S.filter(covered).length} of ${S.length}`, '', 'analysis times with an ATMS overpass; the rest are infrared plus prior'),
    tile('805 / 252', '', 'training and validation scenes (2023 season held out, Milton never seen)'),
  );
  // time series
  drawSeries($('series'), $('tip2'), {
    labels: S.map(s => fmtT(s.time).replace(' Oct ', '/')), atms: S.map(covered),
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
