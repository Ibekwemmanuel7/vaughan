# ---------- Export the explorer fields for a second or third storm (CPU runtime is enough) ----------
# Writes MyDrive/milton_artifacts/dash_<storm>.json in the same per-scene format as the Milton dashboard_data.json
# (scalar scores plus 64 x 64 fields), so dashboard/build.py can pack it and the explorer's storm selector can show it.
# Self-contained: mounts Drive and unpacks melissa_scenes.zip and polo_scenes_era5.zip from milton_da_upload if needed.
# Run once per storm; copy the JSON to dashboard/raw/dash_<storm>.json on the PC and rebuild the site.
"""
import glob, json, os, shutil, subprocess
import numpy as np, xarray as xr
from google.colab import drive
drive.mount('/content/drive')
A = '/content/drive/MyDrive/milton_artifacts'; UP = '/content/drive/MyDrive/milton_da_upload'
for zname, sdir, pat in (('melissa_scenes.zip', '/content/data/melissa/scenes', 'MELISSA_*.nc'), ('polo_scenes_era5.zip', '/content/data/polo/scenes_era5', 'POLO_*.nc')):
    if glob.glob(f'{sdir}/{pat}'): continue
    if not os.path.exists(f'{UP}/{zname}'): print('no', zname, 'in', UP); continue
    os.makedirs(sdir, exist_ok=True); subprocess.run(['unzip', '-o', '-q', f'{UP}/{zname}', '-d', sdir], check=True)
    for f in glob.glob(f'{sdir}/**/{pat}', recursive=True):
        if os.path.dirname(f) != sdir: shutil.move(f, f'{sdir}/' + os.path.basename(f))
    print(zname, '->', len(glob.glob(f'{sdir}/{pat}')), 'scenes')
JOBS = {   # storm: (analysis folder, scene folder, pattern, rain reference label)
    'melissa': (f'{A}/melissa_analysis', '/content/data/melissa/scenes', 'MELISSA_*', 'IMERG Late Run'),
    'polo': (f'{A}/polo_analysis_plain_era5', '/content/data/polo/scenes_era5', 'POLO_*', 'IMERG Late Run'),
}
CORE_HALF = 8
def coarsen(a, n=64):
    a = np.asarray(a, np.float32); f = max(a.shape[-1] // n, 1)
    if f > 1:
        H, W = a.shape[-2:]; a = a[..., : H - H % f, : W - W % f].reshape(*a.shape[:-2], H // f, f, W // f, f).mean((-3, -1))
    return np.round(np.nan_to_num(a, nan=0.0), 2).tolist()
def to_grid(a, shape):
    if a.shape == shape: return a
    f = a.shape[0] // shape[0]
    return a[: a.shape[0] - a.shape[0] % f, : a.shape[1] - a.shape[1] % f].reshape(shape[0], f, shape[1], f).mean((1, 3))
def warm_core(t, radius_px=48):
    L, H, W = t.shape; yy, xx = np.mgrid[:H, :W]; r = np.hypot(yy - H / 2, xx - W / 2)
    return (t[:, H // 2 - 2: H // 2 + 3, W // 2 - 2: W // 2 + 3].mean((-2, -1)) - t[:, r > radius_px].mean(-1))
def rm(a, b, sl=None):
    aa, bb = (a[..., sl, sl], b[..., sl, sl]) if sl is not None else (a, b)
    ok = np.isfinite(aa) & np.isfinite(bb)
    return float(np.sqrt(np.mean((aa[ok] - bb[ok]) ** 2))) if ok.mean() > 0.5 else None
for storm, (RUN, SC, pat, rain_ref) in JOBS.items():
    files = sorted(glob.glob(f'{RUN}/{pat}_analysis.nc'))
    if not files: print(storm, ': no analyses in', RUN); continue
    scenes = []
    for f in files:
        ds = xr.load_dataset(f); sp = f'{SC}/' + os.path.basename(f).replace('_analysis', '')
        if not os.path.exists(sp): print('no scene for', os.path.basename(f)); continue
        sc = xr.load_dataset(sp)
        lev = [int(v) for v in ds['level'].values]; k3, k8 = lev.index(300), lev.index(850)
        t, tu, spd = ds['temperature'].values, ds['temperature_unet'].values, ds['temperature_spread'].values
        H = t.shape[-1]; c = H // 2; s = slice(c - CORE_HALF, c + CORE_HALF)
        rec = {'time': str(ds.attrs.get('time'))[:16], 'levels_hpa': lev, 'ensemble_size': int(ds.attrs.get('ensemble_size', 0)), 'n_steps': int(ds.attrs.get('n_steps', 0)),
               'warm_core_K': np.round(ds['warm_core_anomaly'].values, 2).tolist(), 'warm_core_unet_K': np.round(warm_core(tu), 2).tolist(),
               'mw_coverage': float(sc['mw_mask'].values.mean()), 'spread_300_K': round(float(spd[k3].mean()), 3), 'spread_core_300_K': round(float(spd[k3][s, s].mean()), 3),
               'storm_lat': float(sc.attrs.get('storm_lat', np.nan)), 'storm_lon': float(sc.attrs.get('storm_lon', np.nan)),
               'fields': {'T300': coarsen(t[k3]), 'T850': coarsen(t[k8]), 'precip': coarsen(ds['precip'].values), 'xsec_T': coarsen(t[:, H // 2, :], 128),
                          'ir_obs_C13': coarsen(ds['ir_tb_observed'].sel(ir_channel='C13').values), 'mw_obs_7': coarsen(ds['mw_tb_observed'].sel(mw_channel=7).values, 16),
                          'mw_sim_7': coarsen(ds['mw_tb_simulated'].sel(mw_channel=7).values, 16), 'T300_unet': coarsen(tu[k3]), 'spread300': coarsen(spd[k3])}}
        if 'temp' in sc and np.isfinite(sc['temp'].values).all():
            T = np.stack([to_grid(sc['temp'].values[i], (H, H)) for i in range(len(lev))])
            rec['warm_core_era5_K'] = np.round(warm_core(T), 2).tolist()
            rec['inner_core_rmse_K'] = {'analysis': rm(t[k3], T[k3], s), 'unet': rm(tu[k3], T[k3], s)}
            rec['domain_rmse_K'] = {'analysis': rm(t, T)}
            rec['truth'] = {'T300': coarsen(T[k3]), 'T850': coarsen(T[k8]), 'xsec_T': coarsen(T[:, H // 2, :], 128)}
        else:
            rec['warm_core_era5_K'] = None; rec['inner_core_rmse_K'] = {'analysis': None, 'unet': None}; rec['domain_rmse_K'] = {'analysis': None}; rec['truth'] = {}
        if 'precip' in sc and np.isfinite(sc['precip'].values).all():
            pt = to_grid(sc['precip'].values, ds['precip'].shape); rec['precip_rmse_vs_imerg'] = rm(ds['precip'].values, pt); rec['truth']['precip'] = coarsen(pt)
        else:
            rec['precip_rmse_vs_imerg'] = None
        scenes.append(rec); ds.close(); sc.close()
    out = {'storm': storm, 'scenes': scenes, 'temperature_reference': 'ERA5T' if storm == 'polo' else 'ERA5', 'rain_reference': rain_ref, 'run': os.path.basename(RUN)}
    json.dump(out, open(f'{A}/dash_{storm}.json', 'w'))
    print(f'{storm}: {len(scenes)} scenes, {sum(1 for r in scenes if r["truth"])} with temperature labels -> {A}/dash_{storm}.json ({os.path.getsize(f"{A}/dash_{storm}.json") / 1e6:.1f} MB)')
"""
