"""
Hurricane Polo (EP172026, September 2026) with the Milton-trained system, while the storm is still active.

Third storm, first live one. What is different from Melissa:
  * the track comes from the NHC ATCF b-deck (updated every advisory), not IBTrACS;
  * there is no ERA5 yet (ERA5T lags about five days), so the scenes carry NaN temperature labels and the
    attribute labels="precip" or "none"; the retrieval never reads the labels, only the scoring does;
  * verification is therefore in observation space (simulated minus observed radiances), against IMERG rain
    where it exists, and against the best-track intensity (Vmax, MSLP) as a trend check on the warm core.
    A week from now, rebuild the scenes with --rebuild (no --live) and rerun cell P3 for the ERA5 numbers.
  * Polo sits at 15 to 17 N, 102 to 108 W: outside the GOES-East CONUS sector, so the scenes are cut from the
    GOES-19 full disk (the builder chooses that itself). Polo's satellite zenith angle from GOES-East (75.2 W)
    is about 35 degrees, inside the range of the Atlantic training archive.

Known limits, so they are stated before the numbers: the prior's warm core saturates near 4 K at 300 hPa
(Milton at Category 5 retrieved about 4 K against an ERA5 value that is itself too low; the real core of a
892 hPa storm is well above 10 K). Treat the retrieved warm-core series as an intensity trend indicator, not
as the physical core temperature. The spread is underdispersive (error-to-spread about 12 on Milton).

Scenes are built on the PC (Earthdata login for ATMS and IMERG; no CDS needed for --live):

    cd C:\\Users\\taylo\\milton_da
    python -m vaughan.scripts.prepare_milton --root data\\polo --track atcf --atcf-id EP172026 --storm POLO ^
        --start 2026-09-21T02 --end 2026-09-24T20 --step-hours 6 --goes-satellite goes19 --live
    (02/08/14/20 UTC, not the synoptic hours: at 102 W the sounder overpasses fall near 08:30 and 20:30 UTC,
     and the 90-minute window around 06 and 18 UTC misses them)
    Compress-Archive -Path data\\polo\\scenes\\POLO_*.nc -DestinationPath polo_scenes.zip -Force
    then upload polo_scenes.zip to MyDrive/milton_da_upload/

Runtime: T4 GPU. Run the usual setup cell first (unzips milton_da.zip and the latest patch into /content,
installs xarray netCDF4 h5netcdf pyproj, runs pytest), then cells P1 to P4 in order. Checkpoints, normaliser
and rtm_audit.json are the Milton ones in MyDrive/milton_artifacts: nothing is retrained.
"""

# ---------- CELL P1: unpack the Polo scenes and fetch the b-deck ----------
"""
import glob, os, shutil, requests
UP = '/content/drive/MyDrive/milton_da_upload'
A = '/content/drive/MyDrive/milton_artifacts'
SC = '/content/data/polo/scenes'
os.makedirs(SC, exist_ok=True)
!unzip -o -q $UP/polo_scenes.zip -d $SC
for f in glob.glob(f'{SC}/**/POLO_*.nc', recursive=True):
    if os.path.dirname(f) != SC:
        shutil.move(f, f'{SC}/' + os.path.basename(f))
S = sorted(glob.glob(f'{SC}/POLO_*.nc'))
print(len(S), 'scenes')
import xarray as xr
for s in S:
    ds = xr.open_dataset(s)
    print(os.path.basename(s), '| labels:', ds.attrs.get('labels'), '| ATMS coverage:', round(float(ds['mw_mask'].mean()), 2),
          '| dt', ds.attrs.get('mw_dt_min'), 'min |', ds.attrs.get('atms_files', '')[:28])
    ds.close()
# best track (b-deck) for the intensity panel; re-downloaded each run because it grows while Polo is active
os.makedirs('/content/data/polo/raw/atcf', exist_ok=True)
BD = '/content/data/polo/raw/atcf/bep172026.dat'
open(BD, 'wb').write(requests.get('https://ftp.nhc.noaa.gov/atcf/btk/bep172026.dat', timeout=60).content)
import sys; sys.path.insert(0, '/content')
from vaughan.data.best_track import BestTrack
bt = BestTrack.from_atcf_bdeck(BD)
print(bt.name, bt.sid, len(bt.times), 'fixes; peak', bt.wind_kt.max(), 'kt,', bt.mslp_hpa.min(), 'hPa at', bt.times[int(bt.wind_kt.argmax())])
"""

# ---------- CELL P2: the full retrieval on all scenes (about 4 minutes per scene on a T4) ----------
# Same command as the Milton and Melissa runs: 8 members, 500 steps, small preset at 128 x 128, ATMS 5 to 9
# in the physics likelihood, calibrated from the archive_2023 audit table. Scenes without labels simply get
# no temperature_rmse_vs_era5 variable.
"""
%cd /content
!python -m vaughan.inference.run_milton --scenes /content/data/polo/scenes/POLO_*.nc \
    --stats $A/norm_stats.json --unet $A/checkpoints/unet.pt --score $A/checkpoints/score.pt \
    --out $A/polo_analysis --preset small --downscale 2 --ensemble 8 --steps 500 \
    --rtm-audit $A/rtm_audit.json --audit-table archive_2023
"""

# ---------- CELL P3: score what can be scored now ----------
# 1. Observation-space fit per scene: RMSE of simulated minus observed brightness temperature, GOES C13 and
#    every ATMS channel in the likelihood (the analysis should fit the sounder better than the U-Net does).
# 2. Rain against IMERG where the scene has it: inner-core mean rate and RMSE.
# 3. Warm core at 300 hPa through time (analysis, U-Net) next to the best-track Vmax and MSLP.
# 4. If ERA5 labels exist (after a rebuild), the inner-core RMSE exactly as for Milton and Melissa.
"""
import glob, json, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
CORE_HALF = 8      # central 16 x 16 pixels at the 128 grid, about 70 km
cal = json.load(open(f'{A}/polo_analysis/effective_calibration.json'))
bias = dict(zip([int(c) for c in cal['mw_channels']], cal['mw_bias_K']))

def to_grid(a, shape):
    if a.shape == shape: return a
    f = a.shape[0] // shape[0]
    return a[: a.shape[0] - a.shape[0] % f, : a.shape[1] - a.shape[1] % f].reshape(shape[0], f, shape[1], f).mean((1, 3))

def rm(a, b, sl=None):
    aa, bb = (a[sl, sl], b[sl, sl]) if sl is not None else (a, b)
    ok = np.isfinite(aa) & np.isfinite(bb)
    return float(np.sqrt(np.mean((aa[ok] - bb[ok]) ** 2))) if ok.mean() > 0.5 else np.nan

rows = []
for f in sorted(glob.glob(f'{A}/polo_analysis/POLO_*_analysis.nc')):
    ds = xr.load_dataset(f); sc = xr.load_dataset(f'{SC}/' + os.path.basename(f).replace('_analysis', ''))
    lev = [int(v) for v in ds['level'].values]; k = lev.index(300)
    H = ds['temperature'].shape[-1]; c = H // 2; s = slice(c - CORE_HALF, c + CORE_HALF)
    t = np.datetime64(str(ds.attrs['time'])[:16])
    # best-track intensity interpolated to the analysis time (the scenes sit at 02/08/14/20 UTC, off the six-hourly fixes)
    xb = (bt.times - bt.times[0]) / np.timedelta64(1, 's'); xt = (np.datetime64(t, 'ns') - bt.times[0]) / np.timedelta64(1, 's')
    r = {'time': str(t), 'lat': float(ds.attrs['storm_lat']), 'lon': float(ds.attrs['storm_lon']), 'labels': ds.attrs.get('labels', ''),
         'bt_vmax_kt': float(np.interp(xt, xb, bt.wind_kt)), 'bt_mslp_hpa': float(np.interp(xt, xb, bt.mslp_hpa)),
         'atms': float(sc['mw_mask'].mean()) > 0.05, 'mw_dt_min': float(ds.attrs.get('mw_dt_min', np.nan)),
         'warm_core_300_K': float(ds['warm_core_anomaly'].values[k]), 'warm_core_peak_hpa': int(lev[int(np.argmax(ds['warm_core_anomaly'].values))]),
         'spread_core_300_K': float(ds['temperature_spread'].values[k][s, s].mean()),
         'core_rain_mmh': float(np.nanmean(ds['precip'].values[s, s])), 'max_rain_mmh': float(np.nanmax(ds['precip'].values))}
    # observation-space fit. The infrared channel of the analytic operator is uncalibrated and outside the
    # likelihood, so its raw difference is reported as bias and standard deviation over the valid pixels, not as
    # a fit. The sounder channels are compared after the calibrated bias (effective_calibration.json) is removed,
    # on the calibrated channels only; the RMSE should sit near the sigma assigned to each channel.
    irc = [str(x) for x in ds['ir_channel'].values]
    m = sc['ir_mask'].values; fct = m.shape[0] // H
    mm = m[: m.shape[0] - m.shape[0] % fct, : m.shape[1] - m.shape[1] % fct].reshape(H, fct, H, fct).mean((1, 3)) > 0.5
    r['ir_coverage'] = float(m.mean())
    if 'C13' in irc:
        i = irc.index('C13'); e = (ds['ir_tb_simulated'].values[i] - ds['ir_tb_observed'].values[i])[mm]
        r['c13_bias_K'] = float(e.mean()); r['c13_std_K'] = float(e.std())
    if r['atms']:
        wm = sc['mw_mask'].values.astype(float); h = ds['mw_tb_observed'].shape[-1]; f2 = wm.shape[0] // h
        wmm = wm[: wm.shape[0] - wm.shape[0] % f2, : wm.shape[1] - wm.shape[1] % f2].reshape(h, f2, h, f2).mean((1, 3)) > 0.5
        for ci, ch in enumerate(ds['mw_channel'].values):
            if int(ch) in bias:
                e = (ds['mw_tb_simulated'].values[ci] - ds['mw_tb_observed'].values[ci] - bias[int(ch)])[wmm]; e = e[np.isfinite(e)]
                r[f'fit_atms{int(ch)}_K'] = float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
    if 'precip' in r['labels']:
        pt = to_grid(sc['precip'].values, ds['precip'].shape)
        r['rain_rmse_imerg_mmh'] = rm(ds['precip'].values, pt); r['rain_rmse_imerg_unet_mmh'] = rm(ds['precip_unet'].values, pt)
        r['imerg_core_rain_mmh'] = float(np.nanmean(pt[s, s]))
    if 'temp' in r['labels']:
        tt = to_grid(sc['temp'].values[k], ds['temperature'].shape[-2:])
        r['core_rmse_300_K'] = rm(ds['temperature'].values[k], tt, s); r['core_rmse_300_unet_K'] = rm(ds['temperature_unet'].values[k], tt, s)
    rows.append(r); ds.close(); sc.close()
D = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(D.round(2).to_string(index=False))
os.makedirs(f'{A}/polo_results', exist_ok=True)
D.to_csv(f'{A}/polo_results/polo_scores.csv', index=False)

a = D
on = a[a.atms == True]
print('\\nPolo: %d scenes, %d with an ATMS overpass; labels: %s' % (len(a), len(on), sorted(set(a.labels))))
print('warm core at 300 hPa: mean %.2f K, range %.2f to %.2f K; peak level 250 to 400 hPa in %d of %d scenes'
      % (a.warm_core_300_K.mean(), a.warm_core_300_K.min(), a.warm_core_300_K.max(), int(a.warm_core_peak_hpa.between(250, 400).sum()), len(a)))
print('infrared coverage: min %.2f (scenes below 0.98: %d)' % (a.ir_coverage.min(), int((a.ir_coverage < 0.98).sum())))
fits = [c for c in a.columns if c.startswith('fit_atms')]
if fits: print('ATMS fit after bias removal, overpass scenes (assigned sigma 3.4, 2.3, 1.1, 1.0, 1.3 K):', {c.replace('fit_', '').replace('_K', ''): round(float(on[c].mean()), 2) for c in fits}, 'K')
if 'rain_rmse_imerg_mmh' in a:
    print('rain vs IMERG: analysis RMSE %.2f mm/h (U-Net %.2f); inner-core rate analysis %.1f vs IMERG %.1f mm/h'
          % (a.rain_rmse_imerg_mmh.mean(), a.rain_rmse_imerg_unet_mmh.mean(), a.core_rain_mmh.mean(), a.imerg_core_rain_mmh.mean()))
if 'core_rmse_300_K' in a:
    print('ERA5 inner-core RMSE at 300 hPa: %.2f K (U-Net %.2f)' % (a.core_rmse_300_K.mean(), a.core_rmse_300_unet_K.mean()))
ok = np.isfinite(a.bt_vmax_kt)
if ok.sum() > 3:
    print('rank correlation, retrieved warm core vs best-track Vmax: %.2f; vs MSLP: %.2f'
          % (a.warm_core_300_K[ok].corr(a.bt_vmax_kt[ok], method='spearman'), a.warm_core_300_K[ok].corr(a.bt_mslp_hpa[ok], method='spearman')))

t = pd.to_datetime(a.time)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
ax[0].plot(t, a.warm_core_300_K, 'o-', color='navy', label='analysis warm core, 300 hPa')
ax[0].plot(t[a.atms == True], a.warm_core_300_K[a.atms == True], 'o', color='black', ms=9, mfc='none', mew=1.6, label='ATMS overpass')
ax[0].set_ylabel('core minus environment (K)'); ax[0].set_title('Polo: retrieved warm core'); ax[0].grid(alpha=.3); ax[0].legend(); ax[0].tick_params(axis='x', rotation=30)
bb = pd.DataFrame({'t': bt.times, 'v': bt.wind_kt, 'p': bt.mslp_hpa}); bb = bb[(bb.t >= t.min() - pd.Timedelta('6h')) & (bb.t <= t.max() + pd.Timedelta('6h'))]
ax[1].plot(bb.t, bb.v, 'k-o', ms=3, label='NHC best track Vmax (kt)'); ax[1].set_title('Observed intensity'); ax[1].grid(alpha=.3); ax[1].legend(loc='upper left'); ax[1].tick_params(axis='x', rotation=30)
ax1b = ax[1].twinx(); ax1b.plot(bb.t, bb.p, '-', color='tab:red', label='MSLP (hPa)'); ax1b.legend(loc='upper right')
ax[2].plot(t, a.core_rain_mmh, 's-', color='teal', label='analysis inner-core rain (mm/h)')
if 'imerg_core_rain_mmh' in a: ax[2].plot(t, a.imerg_core_rain_mmh, '--', color='gray', label='IMERG inner-core rain')
ax[2].set_title('Rain'); ax[2].grid(alpha=.3); ax[2].legend(); ax[2].tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig(f'{A}/polo_results/polo_summary.png', dpi=140); plt.show()
print('saved polo_scores.csv and polo_summary.png to', f'{A}/polo_results')
"""

# ---------- CELL P4: the structure pack for one scene (default: the 892 hPa peak, 22 September 18 UTC) ----------
# Four panels: observed GOES C13, retrieved 300 hPa temperature anomaly, retrieved rain with the IMERG
# contour where it exists, and the vertical cross-section of the temperature anomaly through the centre.
# Also writes a compact JSON (polo_structure.json) with the per-scene table for whoever consumes it.
"""
import glob, json, os
import numpy as np, xarray as xr, matplotlib.pyplot as plt
WHEN = '2026-09-22T2000'          # change to any scene stem (22/20 UTC: 155 kt, 892 hPa, near the SNPP overpass)
f = f'{A}/polo_analysis/POLO_{WHEN}_analysis.nc'
ds = xr.load_dataset(f); sc = xr.load_dataset(f'{SC}/POLO_{WHEN}.nc')
lev = [int(v) for v in ds['level'].values]; k = lev.index(300)
T = ds['temperature'].values; H = T.shape[-1]; c = H // 2
env = np.hypot(*np.mgrid[:H, :H] - c) > 0.375 * H
anom = T - np.array([np.nanmean(T[i][env]) for i in range(T.shape[0])])[:, None, None]
irc = [str(x) for x in ds['ir_channel'].values]; i13 = irc.index('C13') if 'C13' in irc else 0
lat, lon = ds['lat'].values, ds['lon'].values
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
m0 = ax[0].pcolormesh(lon, lat, ds['ir_tb_observed'].values[i13], cmap='Greys', vmin=190, vmax=300); ax[0].set_title(f'GOES-19 C13 observed, {WHEN}'); plt.colorbar(m0, ax=ax[0], label='K')
m1 = ax[1].pcolormesh(lon, lat, anom[k], cmap='RdBu_r', vmin=-6, vmax=6); ax[1].set_title('300 hPa temperature anomaly (analysis)'); plt.colorbar(m1, ax=ax[1], label='K')
m2 = ax[2].pcolormesh(lon, lat, ds['precip'].values, cmap='Blues', vmin=0, vmax=30); ax[2].set_title('rain rate (analysis)'); plt.colorbar(m2, ax=ax[2], label='mm/h')
if 'precip' in str(ds.attrs.get('labels', '')):
    pt = sc['precip'].values; f_ = pt.shape[0] // H
    pt = pt[: pt.shape[0] - pt.shape[0] % f_, : pt.shape[1] - pt.shape[1] % f_].reshape(H, f_, H, f_).mean((1, 3))
    ax[2].contour(lon, lat, pt, levels=[5, 15], colors=['k', 'r'], linewidths=0.8)
xs = lon[c, :]
m3 = ax[3].contourf(xs, lev, anom[:, c, :], levels=np.linspace(-6, 6, 13), cmap='RdBu_r'); ax[3].invert_yaxis(); ax[3].set_title('W to E cross-section through the centre'); ax[3].set_ylabel('hPa'); plt.colorbar(m3, ax=ax[3], label='K')
for a_ in ax[:3]: a_.set_xlabel('lon'); a_.set_ylabel('lat')
plt.tight_layout(); plt.savefig(f'{A}/polo_results/polo_structure_{WHEN}.png', dpi=140); plt.show()

import pandas as pd
D = pd.read_csv(f'{A}/polo_results/polo_scores.csv')
keep = [c for c in ['time', 'lat', 'lon', 'bt_vmax_kt', 'bt_mslp_hpa', 'atms', 'mw_dt_min', 'warm_core_300_K', 'warm_core_peak_hpa', 'spread_core_300_K',
                    'core_rain_mmh', 'max_rain_mmh', 'ir_coverage', 'fit_atms7_K', 'fit_atms8_K', 'fit_atms9_K', 'rain_rmse_imerg_mmh', 'core_rmse_300_K'] if c in D]
meta = {'storm': 'POLO', 'atcf_id': 'EP172026', 'system': 'Vaughan (Milton-trained checkpoints, unchanged)', 'inputs': 'GOES-19 ABI infrared, NOAA-20/21 and SNPP ATMS',
        'grid': '128 x 128 at about 4.4 km, storm-centred', 'ensemble': 8, 'steps': 500,
        'caveats': ['warm core magnitude is well below the physical core of a Category 5 storm (Polo peaks near 5.7 K at 300 hPa); use as a trend indicator',
                    'on sounder overpasses the inner-core rain is overestimated against IMERG: the clear-sky microwave operator reads eyewall ice scattering as rain', 'spread is underdispersive (error-to-spread about 12 on Milton)',
                    'no ERA5 verification until ERA5T covers the period; observation-space fit and IMERG rain are the checks available now'],
        'scenes': json.loads(D[keep].to_json(orient='records'))}
json.dump(meta, open(f'{A}/polo_results/polo_structure.json', 'w'), indent=1)
print('wrote', f'{A}/polo_results/polo_structure.json', 'with', len(D), 'scenes')
"""
