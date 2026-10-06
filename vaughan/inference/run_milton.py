"""
End-to-end inference on Hurricane Milton: observations -> U-Net proxy -> physics-guided score-based
posterior sampling -> CF NetCDF analysis with ensemble mean, spread, and physics diagnostics.

Example
-------
python -m vaughan.inference.run_milton \
    --scenes  artifacts/scenes/MILTON_2024-10-08T1200.nc artifacts/scenes/MILTON_2024-10-09T0000.nc \
    --stats   artifacts/norm_stats.json \
    --unet    artifacts/checkpoints/unet.pt --score artifacts/checkpoints/score.pt \
    --out     artifacts/analysis/

Scenes are produced with `data.dataset.build_scene_cache()` from GOES-16 ABI CMIP files,
NOAA-20/SNPP ATMS L1B granules, ERA5 (labels for verification only) and IMERG, storm-centred with
`data.best_track.BestTrack`. At inference ERA5/IMERG are only used for verification metrics.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import xarray as xr

from ..assimilation.guidance import JointLikelihood, Observations
from ..assimilation.sampler import GuidedScoreSampler, SamplerOutput
from ..config import PipelineConfig
from ..data.dataset import HurricaneSceneDataset, Normalizer, collate
from ..models.score_net import ScoreUNet
from ..models.sde import VPSDE
from ..models.unet_xattn import CrossAttentionUNet
from ..physics.constraints import hydrostatic_thickness, warm_core_anomaly
from ..physics.rtm import AnalyticRTM
from ..train.common import get_device, load_checkpoint

log = logging.getLogger("vaughan")


def _block_mean(a: np.ndarray, target_shape) -> np.ndarray:
    """Coarsen a 2D coordinate array to target_shape by block averaging (identity if shapes match)."""
    f = a.shape[0] // target_shape[0]
    if f <= 1:
        return a
    H, W = a.shape
    return a[: H - H % f, : W - W % f].reshape(H // f, f, W // f, f).mean((1, 3))


class RetrievalEngine:
    """Holds all trained components and turns a batch of co-registered observations into an analysis."""

    def __init__(self, cfg: PipelineConfig, norm: Normalizer, unet: CrossAttentionUNet, score_net: ScoreUNet, rtm: Optional[torch.nn.Module] = None, device=None):
        self.cfg, self.norm = cfg, norm
        self.device = device or get_device()
        self.unet = unet.to(self.device).eval()
        self.score_net = score_net.to(self.device).eval()
        d = cfg.data
        self.rtm = (rtm or AnalyticRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)).to(self.device)
        self.likelihood = JointLikelihood(self.rtm, norm, d, cfg.guidance).to(self.device)
        self.sampler = GuidedScoreSampler(self.score_net, VPSDE(cfg.sde), self.likelihood, cfg.guidance)

    @classmethod
    def from_checkpoints(cls, cfg: PipelineConfig, stats_path: str, unet_ckpt: str, score_ckpt: str, device=None, rtm_kind: str = "analytic") -> "RetrievalEngine":
        norm = Normalizer.load(stats_path)
        # a checkpoint trained as an attention-encoder variant records its U-Net config; honour it
        extra = torch.load(unet_ckpt, map_location="cpu").get("extra", {})
        uc = extra.get("unet_cfg")
        if extra.get("ice") and cfg.data.ice == "none":
            cfg.data.ice = extra["ice"]                     # checkpoint trained with ice in the state
        if uc:
            cfg.unet.pos_embed = bool(uc.get("pos_embed", False))
            cfg.unet.self_attn_levels = tuple(uc.get("self_attn_levels", ()))
            cfg.unet.mw_encoder = uc.get("mw_encoder", "grid")
            cfg.unet.graph_k = int(uc.get("graph_k", 16))
            cfg.unet.graph_rounds = int(uc.get("graph_rounds", 3))
        unet = CrossAttentionUNet(cfg.data, cfg.unet)
        load_checkpoint(unet_ckpt, unet)
        score = ScoreUNet(cfg.data, cfg.score)
        ckpt = torch.load(score_ckpt, map_location="cpu")
        score.load_state_dict(ckpt.get("ema", ckpt["model"]))     # prefer EMA weights
        d = cfg.data
        rtm = None
        if rtm_kind == "scattering":
            from ..physics.scatter import ScatteringRTM
            rtm = ScatteringRTM(d.levels_hpa, d.ir_channels, d.mw_channels, d.grid.mw_downscale)
        return cls(cfg, norm, unet, score, rtm=rtm, device=device)

    @torch.no_grad()
    def proxy(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.unet(batch["ir"], batch["ir_mask"], batch["mw"], batch["mw_mask"], batch.get("mw_zen"))

    def analyse(self, batch: Dict[str, torch.Tensor], ensemble_size: Optional[int] = None, log_every: int = 50, proxy_no_mw: bool = False,
                x_prev: Optional[torch.Tensor] = None, dt_prev_h: float = 0.0) -> tuple[SamplerOutput, Observations]:
        """x_prev: the previous analysis of the same storm as a persistence background ([N, B, C, H, W] members or [B, C, H, W]),
        dt_prev_h hours old; used only when cfg.guidance.persist_sigma0 > 0 (see GuidanceConfig)."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if proxy_no_mw:      # experiment: the proxy sees only the infrared; ATMS reaches the state through the physics likelihood alone
            pb = dict(batch); pb["mw"] = torch.zeros_like(batch["mw"]); pb["mw_mask"] = torch.zeros_like(batch["mw_mask"]); pb["mw_zen"] = torch.zeros_like(batch["mw_zen"])
            x_det = self.proxy(pb)
        else:
            x_det = self.proxy(batch)                                                                      # [B, L+1, H, W]
        obs = Observations(x_det, batch["ir_raw"], batch["ir_mask"], batch["mw_raw"], batch["mw_mask"])
        if x_prev is not None:
            obs.x_prev, obs.dt_prev_h = x_prev.to(self.device), float(dt_prev_h)
        out = self.sampler.sample(obs, ensemble_size=ensemble_size, log_every=log_every, callback=lambda d: log.info(f"  step {d['step']:4d} ir_rmse={d['ir_rmse_K']:.2f}K mw_rmse={d['mw_rmse_K']:.2f}K"))
        return out, obs

    def to_dataset(self, out: SamplerOutput, obs: Observations, scene: xr.Dataset, truth: Optional[torch.Tensor] = None) -> xr.Dataset:
        """Package member 0 of the batch as an analysis Dataset in physical units."""
        d = self.cfg.data
        with torch.no_grad():
            t_mem, p_mem = self.norm.state_to_physical(out.samples[:, 0], d.precip_log_transform)         # [N, L, H, W], [N, 1, H, W]
            t_mean, p_mean = t_mem.mean(0), p_mem.mean(0)
            # one member: spread is undefined, not NaN; report zero and flag it in the attributes below
            t_std, p_std = (t_mem.std(0), p_mem.std(0)) if t_mem.shape[0] > 1 else (torch.zeros_like(t_mean), torch.zeros_like(p_mean))
            t_det, p_det = self.norm.state_to_physical(obs.x_det[:1], d.precip_log_transform)
            # ice: transform every member to physical units first, then average (the mean of the normalised
            # state would be a geometric-type mean after the log transform, not the arithmetic ensemble mean)
            ice_members = self.norm.state_ice(out.samples[:, 0], d.levels_hpa)
            ice_mean = {k: v.mean(0, keepdim=True) for k, v in ice_members.items()} if ice_members is not None else None
            ir_sim, mw_sim = self.rtm(t_mean[None], p_mean[None], ice_mean)
            thick = hydrostatic_thickness(t_mean[None], d.levels_hpa)[0]
            wc = warm_core_anomaly(t_mean[None])[0]
        ds = xr.Dataset(
            {
                "temperature": (("level", "y", "x"), t_mean.cpu().numpy(), {"units": "K", "long_name": "analysis ensemble mean temperature"}),
                "temperature_spread": (("level", "y", "x"), t_std.cpu().numpy(), {"units": "K"}),
                "precip": (("y", "x"), p_mean[0].cpu().numpy(), {"units": "mm h-1", "long_name": "analysis ensemble mean precipitation rate"}),
                "precip_spread": (("y", "x"), p_std[0].cpu().numpy(), {"units": "mm h-1"}),
                "temperature_unet": (("level", "y", "x"), t_det[0].cpu().numpy(), {"units": "K", "long_name": "deterministic U-Net proxy"}),
                "precip_unet": (("y", "x"), p_det[0, 0].cpu().numpy(), {"units": "mm h-1"}),
                "layer_thickness": (("layer", "y", "x"), thick.cpu().numpy(), {"units": "m", "long_name": "hypsometric thickness between consecutive levels"}),
                "warm_core_anomaly": (("level",), wc.cpu().numpy(), {"units": "K"}),
                "ir_tb_simulated": (("ir_channel", "y", "x"), ir_sim[0].cpu().numpy(), {"units": "K"}),
                "ir_tb_observed": (("ir_channel", "y", "x"), obs.ir_tb[0].cpu().numpy(), {"units": "K"}),
                "mw_tb_simulated": (("mw_channel", "yc", "xc"), mw_sim[0].cpu().numpy(), {"units": "K"}),
                "mw_tb_observed": (("mw_channel", "yc", "xc"), obs.mw_tb[0].cpu().numpy(), {"units": "K"}),
                "lat": (("y", "x"), _block_mean(scene["lat"].values, t_mean.shape[-2:])), "lon": (("y", "x"), _block_mean(scene["lon"].values, t_mean.shape[-2:])),
            },
            coords={"level": list(d.levels_hpa), "layer": np.arange(d.n_levels - 1), "ir_channel": list(d.ir_channels), "mw_channel": list(d.mw_channels)},
            attrs={**scene.attrs, "ensemble_size": int(out.samples.shape[0]), "n_steps": self.cfg.guidance.n_steps, "spread_defined": int(out.samples.shape[0] > 1),
                   "persist_dt_h": float(obs.dt_prev_h) if obs.x_prev is not None else -1.0,
                   "persist_sigma": float(self.likelihood.persist_var(obs.dt_prev_h) ** 0.5) if (obs.x_prev is not None and self.likelihood.persist_var(obs.dt_prev_h) is not None) else 0.0,
                   "persist_sigma0": float(self.cfg.guidance.persist_sigma0) if obs.x_prev is not None else 0.0,
                   "persist_warm_t": float(self.cfg.guidance.persist_warm_t), "seed": int(getattr(self, "seed", None)) if getattr(self, "seed", None) is not None else -1},
        )
        if ice_mean is not None:
            ds["iwp"] = (("y", "x"), ice_mean["iwp"][0, 0].cpu().numpy(), {"units": "kg m-2", "long_name": "analysis ensemble mean ice water path"})
            ice_det = self.norm.state_ice(obs.x_det[:1], d.levels_hpa)
            ds["iwp_unet"] = (("y", "x"), ice_det["iwp"][0, 0].cpu().numpy(), {"units": "kg m-2"})
            if "ciwc" in ice_mean:
                ds["ciwc"] = (("level", "y", "x"), ice_mean["ciwc"][0].cpu().numpy(), {"units": "kg kg-1", "long_name": "analysis ensemble mean cloud ice water content"})
        if truth is not None:
            t_true, p_true = self.norm.state_to_physical(truth[:1].to(self.device), d.precip_log_transform)
            ds["temperature_rmse_vs_era5"] = (("level",), torch.sqrt(((t_mean - t_true[0]) ** 2).mean((-2, -1))).cpu().numpy())
            ds["precip_rmse_vs_imerg"] = float(torch.sqrt(((p_mean - p_true[0]) ** 2).mean()))
        return ds


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--unet", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ensemble", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--downscale", type=int, default=1, help="2 = run at 128 x 128 (use with checkpoints trained at --downscale 2)")
    ap.add_argument("--preset", choices=["small", "full"], default="full", help="must match the preset the checkpoints were trained with")
    ap.add_argument("--no-mw", action="store_true", help="ablation: zero the ATMS mask so the retrieval is infrared-only")
    ap.add_argument("--no-rtm", action="store_true", help="ablation: drop the radiative-transfer terms from the likelihood")
    ap.add_argument("--proxy-no-mw", action="store_true", help="experiment: hide ATMS from the U-Net proxy; the calibrated physics likelihood alone carries the microwave")
    ap.add_argument("--sigma-unet", type=float, default=None, help="trust in the U-Net proxy (normalised units, default 0.5); raise it to let the physics term compete")
    ap.add_argument("--rtm-audit", default=None, help="rtm_audit.json from colab/audit_rtm_cell.py: per-channel bias correction and error sigma")
    ap.add_argument("--audit-table", default="archive_2023", help="which table in the audit file to calibrate from")
    ap.add_argument("--ir-channels", nargs="*", default=[], help="IR channels kept in the physical likelihood (default none: the U-Net carries the IR)")
    ap.add_argument("--mw-channels", nargs="*", default=["5", "6", "7", "8", "9"], help="ATMS channels kept in the physical likelihood (default: O2 sounding channels)")
    ap.add_argument("--ice", choices=["none", "iwp", "profile"], default=None, help="ice in the state; default: what the U-Net checkpoint was trained with")
    ap.add_argument("--rtm", choices=["analytic", "scattering"], default="analytic", help="scattering = two-stream ice scattering on the microwave channels (physics/scatter.py)")
    ap.add_argument("--scatter-fit", default=None, help="JSON with the per-channel ice-scattering fit from the audit ({channel: {a_K, I0, bias_K, rmse_K}}), or a table name inside --rtm-audit")
    ap.add_argument("--allsky", action="store_true", help="all-sky observation error: downweight cloud-affected pixels with the symmetric cloud predictor instead of dropping channels")
    ap.add_argument("--allsky-slope", type=float, default=None, help="K of extra error per K of symmetric cloud depression (default 0.5)")
    ap.add_argument("--persist", type=float, default=0.0, metavar="SIGMA0",
                    help="time continuity: use the previous analysis of the storm as a background with error sqrt(SIGMA0^2 + q^2 dt_h) "
                         "(normalised units; 0.3 is a reasonable start, the proxy sigma is 0.5). Scenes are processed in time order.")
    ap.add_argument("--persist-q", type=float, default=None, help="error growth per sqrt(hour) for --persist (default 0.15)")
    ap.add_argument("--persist-max-gap", type=float, default=None, help="hours beyond which the previous analysis is not used (default 12)")
    ap.add_argument("--persist-warm-t", type=float, default=None, help="also start the reverse chain from the previous analysis diffused to this t (0 = from noise)")
    ap.add_argument("--persist-ir-prev", type=float, default=1.0, metavar="FACTOR",
                    help="observation-aware background: multiply SIGMA0 by FACTOR when the previous analysis had no sounder coverage "
                         "(an infrared-only analysis is a weaker background; 2 is a reasonable start, 1 = off)")
    ap.add_argument("--persist-mw-now", type=float, default=1.0, metavar="FACTOR",
                    help="observation-aware background: multiply SIGMA0 by FACTOR when the current scene has sounder coverage "
                         "(the sounder should not be pulled toward an older analysis; 2 is a reasonable start, 1 = off)")
    ap.add_argument("--seed", type=int, default=None,
                    help="reproducible sampling: scene i, member k draws its noise from seed + 1000 i + k, so two runs that differ only in "
                         "a flag (e.g. --persist) are paired sample by sample. Recorded in the output attrs.")
    ap.add_argument("--resume", action="store_true",
                    help="skip scenes whose analysis already exists in --out with the same scene checksum (a stopped Colab run picks up "
                         "where it left off; seeds stay paired because they depend on the scene index). With --persist the members of each "
                         "analysis are also saved next to it (<scene>_members.pt) so the chain continues across the restart.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = PipelineConfig()
    if args.preset == "small":
        from ..scripts.train import apply_preset
        apply_preset(cfg, "small")
    if args.steps:
        cfg.guidance.n_steps = args.steps
    if args.sigma_unet is not None:
        cfg.guidance.sigma_unet = args.sigma_unet
    if args.no_rtm:
        cfg.guidance.use_ir_obs, cfg.guidance.use_mw_obs = False, False     # exact zero, immune to the all-sky cap
    cfg.guidance.allsky = bool(args.allsky)
    cfg.guidance.persist_sigma0 = float(args.persist)
    if args.persist_q is not None:
        cfg.guidance.persist_q = args.persist_q
    if args.persist_max_gap is not None:
        cfg.guidance.persist_max_gap_h = args.persist_max_gap
    if args.persist_warm_t is not None:
        cfg.guidance.persist_warm_t = args.persist_warm_t
    for name, v in (("--persist-ir-prev", args.persist_ir_prev), ("--persist-mw-now", args.persist_mw_now)):
        if not (np.isfinite(v) and v > 0):
            ap.error(f"{name} must be a positive finite factor; got {v}")
    if cfg.guidance.persist_warm_t and not (cfg.guidance.final_denoise_t < cfg.guidance.persist_warm_t < 1.0):
        ap.error(f"--persist-warm-t must lie in ({cfg.guidance.final_denoise_t}, 1); got {cfg.guidance.persist_warm_t}")
    if args.allsky_slope is not None:
        cfg.guidance.allsky_slope = args.allsky_slope
    if args.ice is not None:
        cfg.data.ice = args.ice
    engine = RetrievalEngine.from_checkpoints(cfg, args.stats, args.unet, args.score, rtm_kind=args.rtm)
    if args.rtm_audit and not args.no_rtm:
        import json
        table = json.load(open(args.rtm_audit))[args.audit_table]
        engine.likelihood.calibrate_from_audit(table, use_ir=args.ir_channels, use_mw=args.mw_channels)
        lk = engine.likelihood
        log.info("calibrated RTM likelihood from %s [%s]: IR channels %s, MW channels %s (bias K %s, sigma K %s)",
                 args.rtm_audit, args.audit_table, args.ir_channels or "none", args.mw_channels,
                 [round(float(b), 1) for b, w in zip(lk.mw_bias, lk.mw_w) if w > 0],
                 [round(float(w) ** -0.5, 2) for w in lk.mw_w if w > 0])
    if args.scatter_fit:
        import json
        fit = json.load(open(args.rtm_audit))[args.scatter_fit] if (args.rtm_audit and not os.path.exists(args.scatter_fit)) else json.load(open(args.scatter_fit))
        engine.likelihood.calibrate_scatter(fit, use_mw=args.mw_channels)
        log.info("loaded ice-scattering fit for channels %s", [ch for ch in fit if fit[ch].get("fitted")])
    if not args.no_rtm:
        lk = engine.likelihood
        eff = {"mw_channels": list(cfg.data.mw_channels), "mw_bias_K": [float(b) for b in lk.mw_bias], "mw_sigma_K": [float(w) ** -0.5 if w > 0 else None for w in lk.mw_w],
               "ir_channels": list(cfg.data.ir_channels), "ir_bias_K": [float(b) for b in lk.ir_bias], "ir_sigma_K": [float(w) ** -0.5 if w > 0 else None for w in lk.ir_w],
               "audit": args.rtm_audit, "audit_table": args.audit_table, "scatter_fit": args.scatter_fit, "allsky": cfg.guidance.allsky}
        os.makedirs(args.out, exist_ok=True)
        import json
        json.dump(eff, open(os.path.join(args.out, "effective_calibration.json"), "w"), indent=1)
        log.info("effective calibration written to %s (audit first, scatter fit overrides fitted channels)", os.path.join(args.out, "effective_calibration.json"))

    # scenes in storm and time order (needed for --persist; harmless otherwise)
    def scene_time(p):
        with xr.open_dataset(p) as d:
            try:
                return np.datetime64(str(d.attrs.get("time", ""))[:16])
            except ValueError:                                              # synthetic or undated scene
                return np.datetime64("NaT")
    def scene_key(p):
        """(storm name, grid shape, centre resolution) that a persistence background must share with the scene it feeds."""
        with xr.open_dataset(p) as d:
            return (str(d.attrs.get("storm_name", "")), tuple(int(n) for n in d["ir"].shape[-2:]) if "ir" in d else ())
    times = [scene_time(p) for p in args.scenes]
    if all(not np.isnat(t) for t in times):
        # by storm, then by time, so each storm's persistence chain is contiguous even in a mixed list
        args.scenes = [p for _, _, p in sorted(zip([scene_key(p) for p in args.scenes], times, args.scenes), key=lambda z: (z[0], z[1]))]
    elif cfg.guidance.persist_sigma0 > 0:
        bad = [p for p, t in zip(args.scenes, times) if np.isnat(t)]
        raise SystemExit(f"--persist needs a valid attrs['time'] on every scene; missing or unparsable on {bad[:3]}{' ...' if len(bad) > 3 else ''}")
    if args.seed is not None:
        log.info("seed %d: scene i, member k uses seed + 1000 i + k", args.seed)
    ds_obj = HurricaneSceneDataset(args.scenes, cfg.data, engine.norm, downscale=args.downscale)
    os.makedirs(args.out, exist_ok=True)
    prev_samples, prev_time, prev_key = None, None, None
    sigma0_base, prev_had_mw = cfg.guidance.persist_sigma0, False
    if cfg.guidance.persist_sigma0 > 0:
        log.info("time continuity on: sigma_p^2 = %.2f^2 + %.2f^2 dt_h, max gap %.0f h, warm start t = %.2f; sigma0 x %.1f after an IR-only analysis, x %.1f on a sounder scene",
                 cfg.guidance.persist_sigma0, cfg.guidance.persist_q, cfg.guidance.persist_max_gap_h, cfg.guidance.persist_warm_t, args.persist_ir_prev, args.persist_mw_now)
    for i in range(len(ds_obj)):
        scene = xr.load_dataset(args.scenes[i])
        batch = collate([ds_obj[i]])
        if args.no_mw:
            batch["mw_mask"].zero_(), batch["mw"].zero_(), batch["mw_zen"].zero_()
        t_now, key_now = scene_time(args.scenes[i]), scene_key(args.scenes[i])
        x_prev, dt_h = None, 0.0
        if sigma0_base > 0 and prev_samples is not None:
            if key_now != prev_key:
                log.info("scene %s is a different storm or grid (%s vs %s): persistence chain reset", os.path.basename(args.scenes[i]), key_now, prev_key)
            else:
                dt_h = float((t_now - prev_time) / np.timedelta64(1, "h"))
                if 0 <= dt_h <= cfg.guidance.persist_max_gap_h:
                    x_prev = prev_samples
                else:
                    log.info("previous analysis %.1f h old: outside the persistence window, starting fresh", dt_h)
        has_mw = bool(float(batch["mw_mask"].float().mean()) > 0.05)
        path = os.path.join(args.out, os.path.basename(args.scenes[i]).replace(".nc", "_analysis.nc"))
        members_path = path.replace("_analysis.nc", "_members.pt")
        if args.resume and os.path.exists(path):
            done_sha = xr.open_dataset(path).attrs.get("scene_sha256")
            if done_sha == _sha256(args.scenes[i]):
                # already analysed from this exact scene file: keep it, and let the chain continue from its saved members
                prev_samples = torch.load(members_path, map_location="cpu") if os.path.exists(members_path) else None
                prev_time, prev_key, prev_had_mw = t_now, key_now, has_mw
                if sigma0_base > 0 and prev_samples is None:
                    log.info("resume: %s exists but has no saved members; the persistence chain restarts at the next scene", os.path.basename(path))
                else:
                    log.info("resume: %s exists, skipped", os.path.basename(path))
                continue
            log.info("resume: %s exists but was made from a different scene file (checksum differs); redoing it", os.path.basename(path))
        if x_prev is not None:
            # observation-aware background error: the previous analysis counts less when it had no sounder, and the
            # current scene leans on it less when it has one. Both factors 1 gives the plain persistence term.
            f = (args.persist_ir_prev if not prev_had_mw else 1.0) * (args.persist_mw_now if has_mw else 1.0)
            cfg.guidance.persist_sigma0 = sigma0_base * f
            if f != 1.0:
                log.info("background sigma0 %.2f (x %.1f: previous %s, current %s)", cfg.guidance.persist_sigma0, f,
                         "sounder" if prev_had_mw else "IR only", "sounder" if has_mw else "IR only")
        if args.seed is not None:
            engine.seed = args.seed + 1000 * i
            torch.manual_seed(engine.seed)                                # the sampler derives member k from initial_seed + k
        out, obs = engine.analyse(batch, ensemble_size=args.ensemble, proxy_no_mw=args.proxy_no_mw, x_prev=x_prev, dt_prev_h=dt_h)
        prev_samples, prev_time, prev_key, prev_had_mw = out.samples.detach(), t_now, key_now, has_mw
        truth = batch["state"] if bool(torch.isfinite(batch["state"]).all()) else None     # live scenes carry NaN labels: no RMSE
        result = engine.to_dataset(out, obs, scene, truth=truth)
        # provenance: which scene file (by content) and which observations this analysis was made from, so two runs can be
        # checked for identical inputs (the Polo persistence runs of Sept/Oct 2026 silently used a stale scene zip on one day)
        result.attrs.update({"scene_file": os.path.abspath(args.scenes[i]), "scene_sha256": _sha256(args.scenes[i]),
                             "ir_coverage": float(batch["ir_mask"].float().mean()), "mw_coverage": float(batch["mw_mask"].float().mean()),
                             "unet_ckpt": os.path.abspath(args.unet), "score_ckpt": os.path.abspath(args.score), "cli": " ".join(sys.argv[1:])})
        result.to_netcdf(path)
        if args.resume and sigma0_base > 0:
            torch.save(out.samples.detach().cpu(), members_path)        # lets --resume continue the chain after a stopped run
        log.info(f"wrote {path}  warm core (K by level): {np.round(result['warm_core_anomaly'].values, 1)}")


if __name__ == "__main__":
    main()
