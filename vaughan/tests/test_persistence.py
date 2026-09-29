"""Time continuity: the previous analysis of the same storm as a persistence background.

Off by default (persist_sigma0 = 0) the likelihood is unchanged. On, the "prev" term pulls the Tweedie
estimate toward x_prev with a variance that grows with the age of the previous analysis, member k of the
new ensemble is conditioned on member k of the old one, and the chain can optionally start from x_prev
diffused to t = persist_warm_t instead of from pure noise.
"""
import numpy as np
import torch

from vaughan.assimilation.guidance import JointLikelihood, Observations
from vaughan.assimilation.sampler import GuidedScoreSampler
from vaughan.config import PipelineConfig
from vaughan.data.dataset import HurricaneSceneDataset, Normalizer, collate
from vaughan.data.synthetic import make_synthetic_scenes
from vaughan.inference.run_milton import RetrievalEngine
from vaughan.models.score_net import GaussianClimatologyScore, ScoreUNet
from vaughan.models.sde import VPSDE
from vaughan.models.unet_xattn import CrossAttentionUNet
from vaughan.physics.rtm import AnalyticRTM


def _setup(n_scenes=3):
    torch.manual_seed(0)
    cfg = PipelineConfig.small_debug()
    d = cfg.data
    scenes = make_synthetic_scenes(d, n_scenes)
    norm = Normalizer.fit(scenes)
    rtm = AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
    return cfg, d, scenes, norm, rtm


def test_persist_var_schedule():
    cfg, d, scenes, norm, rtm = _setup()
    lik = JointLikelihood(rtm, norm, cfg.data, cfg.guidance)
    assert lik.persist_var(6.0) is None                                    # off by default
    cfg.guidance.persist_sigma0, cfg.guidance.persist_q, cfg.guidance.persist_max_gap_h = 0.3, 0.15, 12.0
    v0, v6, v12 = lik.persist_var(0.0), lik.persist_var(6.0), lik.persist_var(12.0)
    assert abs(v0 - 0.09) < 1e-9 and v0 < v6 < v12 and abs(v6 - (0.09 + 0.0225 * 6)) < 1e-9   # grows linearly in dt
    assert lik.persist_var(12.5) is None and lik.persist_var(-1.0) is None    # outside the window


def test_prev_term_zero_when_off_and_pulls_toward_x_prev_when_on():
    cfg, d, scenes, norm, rtm = _setup()
    batch = collate([HurricaneSceneDataset([scenes[0]], d, norm)[0]])
    lik = JointLikelihood(rtm, norm, cfg.data, cfg.guidance)
    x_det = batch["state"]
    obs = Observations(x_det, batch["ir_raw"], batch["ir_mask"], batch["mw_raw"], batch["mw_mask"])
    x0 = x_det + 0.5
    obs.x_prev, obs.dt_prev_h = x_det - 0.5, 3.0
    terms_off = lik.terms(x0, obs, r2=0.0)
    assert "prev" in terms_off and float(terms_off["prev"]) == 0.0
    cfg.guidance.persist_sigma0 = 0.3
    terms_on = lik.terms(x0, obs, r2=0.0)
    n = x0[0].numel()
    expected = 0.5 * n * 1.0 / lik.persist_var(3.0)                        # (x0 - x_prev)^2 = 1 everywhere
    assert abs(float(terms_on["prev"]) - expected) / expected < 1e-5
    # the other terms are untouched by the new one
    for k in ("unet", "ir", "mw", "stability", "precip_nonneg"):
        assert torch.allclose(terms_on[k], terms_off[k])
    # an older previous analysis counts for less
    obs.dt_prev_h = 9.0
    assert float(lik.terms(x0, obs, r2=0.0)["prev"]) < float(terms_on["prev"])
    # the gradient points from x0 toward x_prev
    x0g = x0.clone().requires_grad_(True)
    g = torch.autograd.grad(lik.terms(x0g, obs, r2=0.0)["prev"].sum(), x0g)[0]
    assert (g > 0).all()                                                    # x0 > x_prev, so descending in x0 moves toward x_prev


def test_for_member_selects_the_matching_previous_member():
    cfg, d, scenes, norm, rtm = _setup()
    batch = collate([HurricaneSceneDataset([scenes[0]], d, norm)[0]])
    obs = Observations(batch["state"], batch["ir_raw"], batch["ir_mask"], batch["mw_raw"], batch["mw_mask"])
    prev = torch.stack([batch["state"] + k for k in range(3)], 0)             # [N=3, B, C, H, W]
    obs.x_prev, obs.dt_prev_h = prev, 2.0
    o1 = obs.for_member(1)
    assert torch.equal(o1.x_prev, prev[1]) and o1.dt_prev_h == 2.0 and o1.x_prev.dim() == 4
    assert torch.equal(obs.for_member(5).x_prev, prev[5 % 3])              # more members than before: wrap around
    obs.x_prev = prev[0]                                                    # a single previous field serves every member
    assert torch.equal(obs.for_member(2).x_prev, prev[0])
    assert Observations(batch["state"], batch["ir_raw"], batch["ir_mask"], batch["mw_raw"], batch["mw_mask"]).for_member(0).x_prev is None


def test_persistence_moves_the_analysis_toward_the_previous_one():
    """Exact Gaussian prior, no observations to speak of (huge sigmas): the analysis with a persistence
    background must sit closer to x_prev than the analysis without it."""
    cfg, d, scenes, norm, rtm = _setup()
    g = cfg.guidance
    g.n_steps, g.ensemble_size, g.n_corrector = 40, 1, 1
    g.sigma_ir_K, g.sigma_mw_K, g.sigma_unet = 1e3, 1e3, 1e3
    batch = collate([HurricaneSceneDataset([scenes[0]], d, norm)[0]])
    sde = VPSDE(cfg.sde)
    prior = GaussianClimatologyScore.fit_from_dataset(HurricaneSceneDataset(scenes, d, norm), sde)
    lik = JointLikelihood(rtm, norm, cfg.data, g)
    sampler = GuidedScoreSampler(prior, sde, lik, g)
    obs = Observations(batch["state"], batch["ir_raw"], batch["ir_mask"], batch["mw_raw"], batch["mw_mask"])
    x_prev = batch["state"] + 1.5                                           # a deliberately displaced "previous analysis"

    torch.manual_seed(1)
    free = sampler.sample(obs).mean
    g.persist_sigma0, g.persist_q = 0.2, 0.0
    obs.x_prev, obs.dt_prev_h = x_prev, 3.0
    torch.manual_seed(1)
    held = sampler.sample(obs).mean
    d_free, d_held = float((free - x_prev).pow(2).mean()), float((held - x_prev).pow(2).mean())
    assert d_held < 0.5 * d_free, (d_free, d_held)

    # warm start from x_prev diffused to t = 0.5: shorter chain, still finite, still close to x_prev
    g.persist_warm_t = 0.5
    torch.manual_seed(1)
    warm, trace = sampler.sample_chain(obs)
    assert torch.isfinite(warm).all() and trace[0]["t"] <= 0.5 + 1e-6
    assert float((warm - x_prev).pow(2).mean()) < 0.5 * d_free


def test_engine_threads_x_prev_and_records_attrs():
    cfg, d, scenes, norm, rtm = _setup()
    cfg.guidance.n_steps, cfg.guidance.persist_sigma0 = 4, 0.3
    engine = RetrievalEngine(cfg, norm, CrossAttentionUNet(d, cfg.unet), ScoreUNet(d, cfg.score), rtm, device=torch.device("cpu"))
    b0 = collate([HurricaneSceneDataset([scenes[0]], d, norm)[0]])
    b1 = collate([HurricaneSceneDataset([scenes[1]], d, norm)[0]])
    out0, obs0 = engine.analyse(b0, ensemble_size=2)
    ds0 = engine.to_dataset(out0, obs0, scenes[0], truth=b0["state"])
    assert ds0.attrs["persist_dt_h"] == -1.0 and ds0.attrs["persist_sigma"] == 0.0
    out1, obs1 = engine.analyse(b1, ensemble_size=2, x_prev=out0.samples, dt_prev_h=6.0)
    assert obs1.x_prev is not None and obs1.x_prev.shape[0] == 2 and torch.isfinite(out1.samples).all()
    ds1 = engine.to_dataset(out1, obs1, scenes[1], truth=b1["state"])
    assert ds1.attrs["persist_dt_h"] == 6.0
    assert abs(ds1.attrs["persist_sigma"] - np.sqrt(0.09 + 0.15**2 * 6)) < 1e-6
