"""
Hurricane Isaias (AL092026, October 2026) with the Milton-trained system: the fourth storm.

Why this one: it is the hard direction for the claims on the dashboard. Milton, Melissa and Polo were intense,
symmetric storms; Isaias was a sheared Gulf storm that became the season's first hurricane on 8 October and
made landfall on the Florida Panhandle on 9 October, so its warm core is shallow and tilted and the infrared
cloud top is a poor guide to it. It sits inside the GOES-East CONUS sector and the Atlantic training archive,
like Milton, so the prior is on home ground. It is run after the fact: ERA5T covers October by now.

Scenes are built on the PC (Earthdata login for ATMS and IMERG, CDS key for ERA5):

    cd C:\\Users\\taylo\\milton_da
    python -m vaughan.scripts.prepare_milton --root data\\isaias --storm ISAIAS --season 2026 --basin NA ^
        --start 2026-10-07T00 --end 2026-10-10T12 --step-hours 6 --goes-satellite goes19 --dry-run
    (check the track covers the window and the times; then run the same line without --dry-run,
     and once more with --start 2026-10-07T02 --end 2026-10-10T08: over the Gulf near 88 W the sounder
     passes near 07:30 and 19:30 UTC, so the 02/08/14/20 grid catches the overpasses the synoptic grid misses)
    If IBTrACS does not have Isaias yet, use the NHC b-deck instead: --track atcf --atcf-id AL092026
    Compress-Archive -Path data\\isaias\\scenes\\ISAIAS_*.nc -DestinationPath isaias_scenes.zip -Force
    then upload isaias_scenes.zip to MyDrive/milton_da_upload/

Runtime: T4 GPU (about 4 minutes per scene; 8 members, 500 steps). Run the setup cell of vaughan_polo.ipynb
first (it unpacks the package and mounts Drive), then I1 to I4. I2 resumes if the session stops: rerun it.
"""

# ---------- CELL I1: unpack the Isaias scenes ----------
"""
import glob, os, shutil
import numpy as np, xarray as xr
UP = '/content/drive/MyDrive/milton_da_upload'; A = '/content/drive/MyDrive/milton_artifacts'
SC = '/content/data/isaias/scenes'; os.makedirs(SC, exist_ok=True)
!unzip -o -q $UP/isaias_scenes.zip -d $SC
for f in glob.glob(f'{SC}/**/ISAIAS_*.nc', recursive=True):
    if os.path.dirname(f) != SC: shutil.move(f, f'{SC}/' + os.path.basename(f))
S = sorted(glob.glob(f'{SC}/ISAIAS_*.nc')); print(len(S), 'scenes')
for s in S:
    ds = xr.open_dataset(s)
    print(os.path.basename(s), '| ATMS', round(float(ds['mw_mask'].mean()), 2), '| ERA5', 'temp' in ds and bool(np.isfinite(ds['temp'].values).all()),
          '| IMERG', 'precip' in ds and bool(np.isfinite(ds['precip'].values).all()), '|', ds.attrs.get('storm_lat'), ds.attrs.get('storm_lon'))
    ds.close()
"""

# ---------- CELL I2: the retrieval, unchanged, paired seeds, resumable ----------
"""
%cd /content
!python -m vaughan.inference.run_milton --scenes /content/data/isaias/scenes/ISAIAS_*.nc \
    --stats $A/norm_stats.json --unet $A/checkpoints/unet.pt --score $A/checkpoints/score.pt \
    --out $A/isaias_analysis --preset small --downscale 2 --ensemble 8 --steps 500 \
    --rtm-audit $A/rtm_audit.json --audit-table archive_2023 --seed 7 --resume
"""

# ---------- CELL I3: score against ERA5 and IMERG (CPU is enough) ----------
# Same definitions as Milton, Melissa and Polo: inner core = central 16 x 16 px (about 70 km) at 300 hPa; domain = all
# ten levels on the full grid; warm core = 5 x 5 centre minus the environment beyond 48 px. Scenes after landfall are
# scored but marked: the training archive is over water, so those are out of distribution by construction.
"""
import glob, json, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
A = '/content/drive/MyDrive/milton_artifacts'; SC = '/content/data/isaias/scenes'
CORE_HALF = 8
def to_grid(a, shape):
    if a.shape == shape: return a
    f = a.shape[0] // shape[0]
    return a[: a.shape[0] - a.shape[0] % f, : a.shape[1] - a.shape[1] % f].reshape(shape[0], f, shape[1], f).mean((1, 3))
def warm_core(t, radius_px=48):
    H, W = t.shape; yy, xx = np.mgrid[:H, :W]; r = np.hypot(yy - H / 2, xx - W / 2)
    return float(t[H // 2 - 2: H // 2 + 3, W // 2 - 2: W // 2 + 3].mean() - t[r > radius_px].mean())
def rm(a, b, sl=None):
    aa, bb = (a[..., sl, sl], b[..., sl, sl]) if sl is not None else (a, b)
    ok = np.isfinite(aa) & np.isfinite(bb)
    return float(np.sqrt(np.mean((aa[ok] - bb[ok]) ** 2))) if ok.mean() > 0.5 else np.nan
rows = []
for f in sorted(glob.glob(f'{A}/isaias_analysis/ISAIAS_*_analysis.nc')):
    ds = xr.load_dataset(f); sc = xr.load_dataset(f'{SC}/' + os.path.basename(f).replace('_analysis', ''))
    if 'temp' not in sc or not np.isfinite(sc['temp'].values).all(): continue
    lev = [int(v) for v in ds['level'].values]; k = lev.index(300)
    H = ds['temperature'].shape[-1]; c = H // 2; s = slice(c - CORE_HALF, c + CORE_HALF)
    T = np.stack([to_grid(sc['temp'].values[i], (H, H)) for i in range(len(lev))])
    ta, tu = ds['temperature'].values, ds['temperature_unet'].values
    r = {'time': np.datetime64(str(ds.attrs['time'])[:16]), 'atms': float(sc['mw_mask'].mean()) > 0.05,
         'core_rmse_300_K': rm(ta[k], T[k], s), 'core_rmse_300_unet_K': rm(tu[k], T[k], s), 'domain_rmse_all_K': rm(ta, T), 'domain_rmse_all_unet_K': rm(tu, T),
         'wc300_K': warm_core(ta[k]), 'wc300_unet_K': warm_core(tu[k]), 'wc300_era5_K': warm_core(T[k]), 'peak_hPa': lev[int(np.argmax(ds['warm_core_anomaly'].values))],
         'spread_core_300_K': float(ds['temperature_spread'].values[k][s, s].mean())}
    if 'precip' in sc and np.isfinite(sc['precip'].values).all():
        pt = to_grid(sc['precip'].values, ds['precip'].shape); r['rain_rmse_imerg_mmh'] = rm(ds['precip'].values, pt)
    rows.append(r); ds.close(); sc.close()
E = pd.DataFrame(rows).sort_values('time').set_index('time')
g = E[E.atms]
print(f"Isaias n={len(E)} ({len(g)} with ATMS)  inner-core 300 hPa RMSE vs ERA5: sounder scenes {g.core_rmse_300_K.mean():.2f} K (U-Net {g.core_rmse_300_unet_K.mean():.2f}), "
      f"all {E.core_rmse_300_K.mean():.2f} K (U-Net {E.core_rmse_300_unet_K.mean():.2f});  domain {E.domain_rmse_all_K.mean():.2f} K (U-Net {E.domain_rmse_all_unet_K.mean():.2f});  "
      f"warm core bias {(E.wc300_K - E.wc300_era5_K).mean():+.2f} K, rank corr {spearmanr(E.wc300_K, E.wc300_era5_K).correlation:.2f};  peaks at 250-400 hPa: {E.peak_hPa.isin([250, 300, 400]).sum()} of {len(E)}")
print('Compare: Milton 1.13 / 1.77 / 1.87 K, Melissa 0.91 / 1.15 / 0.93 K, Polo 1.20 / 1.26 / 1.38 K (sounder scenes / all scenes / domain)')
print(E.round(2).to_string())
os.makedirs(f'{A}/isaias_results', exist_ok=True); E.to_csv(f'{A}/isaias_results/isaias_era5_scores.csv')
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(E.index, E.wc300_era5_K, 'k-', lw=2, label='ERA5'); ax[0].plot(E.index, E.wc300_K, 'o-', color='C0', label='analysis'); ax[0].plot(E.index, E.wc300_unet_K, 's:', color='C1', label='direct U-Net')
ax[0].set_ylabel('300 hPa warm core (K)'); ax[0].legend(fontsize=8); ax[0].set_title('Hurricane Isaias against ERA5')
ax[1].plot(E.index, E.core_rmse_300_K, 'o-', color='C0', label='analysis'); ax[1].plot(E.index, E.core_rmse_300_unet_K, 's:', color='C1', label='direct U-Net')
ax[1].set_ylabel('inner-core RMSE (K)'); ax[1].legend(fontsize=8)
for t in E.index[E.atms]: ax[1].axvline(t, color='0.85', lw=0.8, zorder=0)
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f'{A}/isaias_results/isaias_summary.png', dpi=150); print('saved isaias_summary.png and isaias_era5_scores.csv')
"""

# ---------- CELL I4: export the explorer fields ----------
# Run colab/export_storm_fields_cell.py (it includes Isaias); copy dash_isaias.json to dashboard/raw/ and rebuild the site.
