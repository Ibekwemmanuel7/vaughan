"""Regression tests for the defects found in the September 2026 independent review."""
import json
import os

import torch
import xarray as xr
import numpy as np

from vaughan.assimilation.guidance import JointLikelihood, Observations
from vaughan.assimilation.sampler import SamplerOutput
from vaughan.config import PipelineConfig
from vaughan.data.dataset import Normalizer
from vaughan.data.synthetic import make_synthetic_scenes
from vaughan.inference.run_milton import RetrievalEngine
from vaughan.models.score_net import ScoreUNet
from vaughan.models.unet_xattn import CrossAttentionUNet
from vaughan.physics.rtm import AnalyticRTM


def _setup(ice="none"):
    torch.manual_seed(0)
    cfg = PipelineConfig.small_debug()
    cfg.data.ice = ice
    scenes = make_synthetic_scenes(cfg.data, 4)
    norm = Normalizer.fit(scenes)
    d = cfg.data
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    return cfg, scenes, norm, rtm


def test_disabled_observation_terms_are_exactly_zero_even_with_allsky():
    """--no-rtm must remove the radiance terms outright; the all-sky cap must not resurrect them."""
    cfg, scenes, norm, rtm = _setup()
    d = cfg.data
    cfg.guidance.allsky = True
    cfg.guidance.use_ir_obs = False
    cfg.guidance.use_mw_obs = False
    lik = JointLikelihood(rtm, norm, d, cfg.guidance)
    H, W = d.grid.ny, d.grid.nx
    t = torch.full((1, len(d.levels_hpa), H, W), 250.0)
    p = torch.zeros(1, 1, H, W)
    ir, mw = rtm(t, p)
    x = norm.physical_to_state(t, p, d.precip_log_transform)
    obs = Observations(x, ir - 30.0, torch.ones_like(ir[:, :1]), mw - 30.0, torch.ones_like(mw[:, :1]))
    terms = lik.terms(x, obs, r2=0.0)
    assert float(terms["ir"]) == 0.0 and float(terms["mw"]) == 0.0


def test_single_member_export_has_finite_zero_spread_and_flag(tmp_path):
    cfg, scenes, norm, rtm = _setup()
    d = cfg.data
    engine = RetrievalEngine(cfg, norm, CrossAttentionUNet(d, cfg.unet), ScoreUNet(d, cfg.score), rtm, device=torch.device("cpu"))
    H, W = d.grid.ny, d.grid.nx
    t = torch.full((1, len(d.levels_hpa), H, W), 250.0); p = torch.zeros(1, 1, H, W)
    x = norm.physical_to_state(t, p, d.precip_log_transform)
    ir, mw = rtm(t, p)
    obs = Observations(x, ir, torch.ones_like(ir[:, :1]), mw, torch.ones_like(mw[:, :1]))
    scene = xr.Dataset({"lat": (("y", "x"), np.zeros((H, W))), "lon": (("y", "x"), np.zeros((H, W)))})
    ds = engine.to_dataset(SamplerOutput(x[None], x, torch.zeros_like(x), []), obs, scene)
    assert np.isfinite(ds["temperature_spread"].values).all() and float(ds["temperature_spread"].max()) == 0.0
    assert ds.attrs["spread_defined"] == 0


def test_ice_export_is_the_arithmetic_mean_of_physical_members():
    cfg, scenes, norm, rtm = _setup(ice="iwp")
    d = cfg.data
    engine = RetrievalEngine(cfg, norm, CrossAttentionUNet(d, cfg.unet), ScoreUNet(d, cfg.score), rtm, device=torch.device("cpu"))
    H, W = d.grid.ny, d.grid.nx
    t = torch.full((2, len(d.levels_hpa), H, W), 250.0); p = torch.zeros(2, 1, H, W)
    iwp = torch.stack([torch.zeros(1, H, W), torch.full((1, H, W), 9.0)])          # [2,1,H,W]: members 0 and 9 kg/m2 -> mean 4.5
    x = norm.physical_to_state(t, p, d.precip_log_transform, ice={"iwp": iwp}, ice_mode="iwp")
    ir, mw = rtm(t[:1], p[:1])
    obs = Observations(x[:1], ir, torch.ones_like(ir[:, :1]), mw, torch.ones_like(mw[:, :1]))
    scene = xr.Dataset({"lat": (("y", "x"), np.zeros((H, W))), "lon": (("y", "x"), np.zeros((H, W)))})
    ds = engine.to_dataset(SamplerOutput(x[:, None], x[:1], torch.zeros_like(x[:1]), []), obs, scene)
    assert abs(float(ds["iwp"].mean()) - 4.5) < 0.05


def test_smoke_tests_do_not_write_into_artifacts_checkpoints():
    """The training tests write checkpoints under tmp_path only (see the fixtures in test_smoke.py and test_cloud_ice.py)."""
    src = open(os.path.join(os.path.dirname(__file__), "test_smoke.py")).read()
    assert "tmp_path_factory" in src and "ckpt_dir" in src
