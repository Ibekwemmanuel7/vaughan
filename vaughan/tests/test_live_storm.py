"""A live storm: ATCF b-deck track, scenes without ERA5 or IMERG labels, and a retrieval that does not read them.

The b-deck lines are real fixes from NHC's bep172026.dat (Hurricane Polo, September 2026), including the
three wind-radius rows per fix and the pre-name INVEST rows.
"""
import numpy as np
import torch
import xarray as xr

from vaughan.config import PipelineConfig
from vaughan.data.best_track import BestTrack
from vaughan.data.dataset import HurricaneSceneDataset, Normalizer, RawScenePaths, collate
from vaughan.data.download import assemble_manifest, goes_sector_for
from vaughan.data.synthetic import make_synthetic_scenes
from vaughan.inference.run_milton import RetrievalEngine
from vaughan.models.score_net import ScoreUNet
from vaughan.models.unet_xattn import CrossAttentionUNet
from vaughan.physics.rtm import AnalyticRTM

BDECK = """EP, 17, 2026092012, , BEST, 0, 140N, 1055W, 30, 1007, DB, 0, , 0, 0, 0, 0, 1011, 180, 90, 40, 0, E, 0, , 0, 0, INVEST, S, 0, , 0, 0, 0, 0, genesis-num, 025,
EP, 17, 2026092018, , BEST, 0, 145N, 1051W, 30, 1007, TD, 0, , 0, 0, 0, 0, 1012, 180, 40, 40, 0, E, 0, , 0, 0, SEVENTEEN, S, 0, , 0, 0, 0, 0, genesis-num, 025, TRANSITIONED, epB92026 to ep172026,
EP, 17, 2026092100, , BEST, 0, 151N, 1048W, 35, 1004, TS, 34, NEQ, 40, 40, 0, 0, 1010, 150, 40, 45, 0, E, 0, , 0, 0, POLO, S, 0, , 0, 0, 0, 0, genesis-num, 025,
EP, 17, 2026092218, , BEST, 0, 146N, 1015W, 155, 892, HU, 34, NEQ, 70, 80, 80, 70, 1008, 130, 10, 190, 10, E, 0, , 0, 0, POLO, D, 12, NEQ, 120, 150, 120, 120, genesis-num, 025,
EP, 17, 2026092218, , BEST, 0, 146N, 1015W, 155, 892, HU, 50, NEQ, 40, 50, 50, 40, 1008, 130, 10, 190, 10, E, 0, , 0, 0, POLO, D, 12, NEQ, 120, 150, 120, 120, genesis-num, 025,
EP, 17, 2026092218, , BEST, 0, 146N, 1015W, 155, 892, HU, 64, NEQ, 25, 30, 30, 25, 1008, 130, 10, 190, 10, E, 0, , 0, 0, POLO, D, 12, NEQ, 120, 150, 120, 120, genesis-num, 025,
EP, 17, 2026092500, , BEST, 0, 171N, 1061W, 130, 938, HU, 34, NEQ, 110, 90, 90, 90, 1006, 130, 15, 0, 0, E, 0, , 0, 0, POLO, D, 0, , 0, 0, 0, 0, genesis-num, 025,
"""


def test_bdeck_parsing_dedupes_radii_rows_and_interpolates(tmp_path):
    p = tmp_path / "bep172026.dat"
    p.write_text(BDECK)
    bt = BestTrack.from_atcf_bdeck(str(p))
    assert len(bt.times) == 5 and bt.sid == "EP172026" and bt.name == "POLO"      # 7 rows, 3 of them radii duplicates
    assert bt.lats[4] == 17.1 and bt.lons[4] == -106.1
    assert bt.wind_kt[3] == 155 and bt.mslp_hpa[3] == 892 and bt.status == ["DB", "TD", "TS", "HU", "HU"]
    lat, lon = bt.center_at(np.datetime64("2026-09-20T15:00"))
    assert abs(lat - 14.25) < 1e-9 and abs(lon + 105.3) < 1e-9
    # Polo's position is served from the GOES-East full disk, Milton's from the CONUS sector, GOES-West always full disk
    assert goes_sector_for(15.8, -102.3, "goes19") == "ABI-L2-CMIPF"
    assert goes_sector_for(16.9, -104.9, "goes19") == "ABI-L2-CMIPF"     # 24 Sept: the CONUS sector cut this domain to 89 percent
    assert goes_sector_for(22.0, -91.0, "goes16") == "ABI-L2-CMIPC"
    assert goes_sector_for(22.0, -91.0, "goes18") == "ABI-L2-CMIPF"


def test_manifest_keeps_unlabelled_scenes_only_when_asked():
    times = [np.datetime64("2026-09-22T18:00"), np.datetime64("2026-09-23T00:00")]
    centers = [(14.6, -101.5), (14.6, -101.4)]
    goes = [{"C13": "a.nc"}, {"C13": "b.nc"}]
    atms = [(["g1.nc"], 12.0), ([], None)]
    strict = assemble_manifest(times, centers, goes, atms, {}, [None, "im.nc"], storm_name="POLO")
    assert strict == []
    live = assemble_manifest(times, centers, goes, atms, {}, [None, "im.nc"], storm_name="POLO", require_labels=False)
    assert len(live) == 2 and live[0].era5_file is None and live[0].imerg_file is None and live[1].imerg_file == "im.nc"


def _unlabel(scene: xr.Dataset) -> xr.Dataset:
    s = scene.copy(deep=True)
    s["temp"].values[:] = np.nan
    s["precip"].values[:] = np.nan
    s.attrs["labels"] = "none"
    return s


def test_retrieval_runs_on_unlabelled_scene_and_skips_rmse():
    torch.manual_seed(0)
    cfg = PipelineConfig.small_debug()
    d = cfg.data
    scenes = make_synthetic_scenes(d, 3)
    norm = Normalizer.fit(scenes)
    live = _unlabel(scenes[0])
    sample = HurricaneSceneDataset([live], d, norm)[0]
    assert torch.isnan(sample["state"]).all() and torch.isfinite(sample["ir"]).all() and torch.isfinite(sample["mw"]).all()
    batch = collate([sample])
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    engine = RetrievalEngine(cfg, norm, CrossAttentionUNet(d, cfg.unet), ScoreUNet(d, cfg.score), rtm, device=torch.device("cpu"))
    engine.cfg.guidance.n_steps = 4
    out, obs = engine.analyse(batch, ensemble_size=2)
    assert torch.isfinite(out.samples).all(), "NaN labels must not reach the sampler"
    truth = batch["state"] if bool(torch.isfinite(batch["state"]).all()) else None
    ds = engine.to_dataset(out, obs, live, truth=truth)
    assert "temperature_rmse_vs_era5" not in ds and "precip_rmse_vs_imerg" not in ds
    assert ds.attrs["labels"] == "none" and np.isfinite(ds["warm_core_anomaly"].values).all()
    # a labelled scene through the same path still gets its RMSE
    batch2 = collate([HurricaneSceneDataset([scenes[1]], d, norm)[0]])
    out2, obs2 = engine.analyse(batch2, ensemble_size=1)
    ds2 = engine.to_dataset(out2, obs2, scenes[1], truth=batch2["state"])
    assert "temperature_rmse_vs_era5" in ds2
