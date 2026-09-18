# Vaughan: physics-guided score-based data assimilation for hurricane structure

Vaughan is the platform and the Python package (`vaughan`, in the `vaughan/` folder of this repository). The repository is `github.com/Ibekwemmanuel7/vaughan`; the earlier address `milton_da` redirects there, and the old dashboard link forwards to the new one. The name is borrowed from The Displacements (Bruce Holsinger, 2022), a novel about a hurricane and the people it uproots.

## Hurricane Milton (Oct 2024) and Hurricane Melissa (Oct 2025)

A modular PyTorch pipeline that reconstructs the 3D temperature structure (200 to 1000 hPa) and 2D
surface precipitation of Hurricane Milton from GOES-16 infrared imagery and ATMS microwave sounder
data, using a Cross-Attention U-Net as a deterministic multi-modal proxy and a conditional
score-based diffusion model, guided by a differentiable radiative-transfer operator, as the
generative inverse solver (the data-assimilation vehicle).

```
observations                     deterministic proxy                 generative DA
GOES ABI  [B,3,256,256] ─┐                                     prior score s_theta(x_t,t)
                         ├─ CrossAttentionUNet ─► x_det ──┐      (ScoreUNet, trained on ERA5/IMERG)
ATMS      [B,9,32,32]  ─┘        (IR queries attend       │              │
                                  to MW tokens)           ▼              ▼
                                            JointLikelihood  ◄──  GuidedScoreSampler (PC, DPS)
                                            = U-Net proxy term          │
                                            + RTM(x) vs TB_obs          ▼
                                            + static stability     posterior ensemble
                                                                   T [N,10,256,256], P [N,1,256,256]
```

## Repository layout

```
vaughan/                   the repository
  vaughan/                 the Python package: import vaughan; python -m vaughan.<module>
  colab/                   Colab cells (training, experiments, Melissa, WeatherNext, cloud ice)
  dashboard/               the published page (index.html) and the logo
  docs/                    the technical report
  results/                 scored outputs (Melissa, WeatherNext)
  data/                    downloaded scenes and raw files (ignored by git)
```

Run everything from the repository root, for example `python -m vaughan.scripts.prepare_milton --root data/milton --dry-run` and `python -m pytest -q vaughan/tests`.

## Package layout (inside `vaughan/`)

| module | contents |
|---|---|
| `config.py` | all dataclass configs, tensor layout conventions, `PipelineConfig.small_debug()` for CPU tests |
| `data/coregistration.py` | `TargetGrid` (storm-centred lat/lon), exact GOES fixed-grid inverse projection sampling, ATMS swath scattered interpolation with coverage masks, ERA5/IMERG regridding |
| `data/dataset.py` | `build_scene()` raw files to one CF xarray scene, `Normalizer` (fit/save/load, differentiable state to physical conversion), `HurricaneSceneDataset`, `collate` |
| `data/best_track.py` | IBTrACS / HURDAT2 centre interpolation for storm-centred domains |
| `data/synthetic.py` | analytic TC scenes with RTM-consistent observations for tests and dry runs |
| `models/blocks.py` | ResBlock (FiLM time conditioning), `CrossAttention2D` (masked, positional, zero-gated), `SelfAttention2D`, time embedding |
| `models/unet_xattn.py` | `CrossAttentionUNet`: IR encoder/decoder, `MicrowaveContextEncoder`, cross-attention at the two deepest levels of both paths |
| `models/score_net.py` | `ScoreUNet` (eps-parameterised prior score), `GaussianClimatologyScore` (closed-form static-B baseline, also used to unit test the sampler) |
| `models/sde.py` | `VPSDE`: marginals, perturbation, Tweedie, DSM loss |
| `physics/rtm.py` | `AnalyticRTM` (weighting-function O2 sounding, rain- or ice-driven scattering depression, IR cloud-top proxy), `NeuralRTMResidual`, `HybridRTM` |
| `physics/scatter.py` | `ScatteringRTM`: Mie soft-sphere ice optics and a delta-Eddington two-stream adding solver for the microwave channels |
| `physics/audit.py` | least-squares fit of the per-channel ice-scattering depression from the operator audit |
| `physics/constraints.py` | static-stability (dry adiabatic) penalty, hypsometric thickness, warm-core diagnostic |
| `assimilation/guidance.py` | `Observations`, `JointLikelihood` (sum-form log p(y given x)), diagnostics |
| `assimilation/sampler.py` | `GuidedScoreSampler`: predictor-corrector on the posterior score, DPS gradient through Tweedie, ensemble output |
| `train/` | `train_unet.py` (MSE + RTM consistency), `train_score.py` (DSM with EMA; optional RTM residual fit), `common.py` |
| `inference/run_milton.py` | `RetrievalEngine` and the CLI that writes CF NetCDF analyses with ensemble mean, spread, simulated vs observed TB, thickness and warm-core anomaly |
| `scripts/` | `prepare_archive`, `prepare_milton`, `train`, `eval_unet`, `add_cloud_ice` command-line entry points |
| `tests/` | end-to-end CPU tests (`test_smoke.py`, `test_cloud_ice.py`, `test_download_selection.py`), 21 in all |

## Tensor conventions

State `x` is `[B, L+1, H, W]`: channels `0..L-1` are temperature at `PRESSURE_LEVELS_HPA`
(200, 250, 300, 400, 500, 600, 700, 850, 925, 1000), channel `L` is `log1p(precip mm/h)`; all
normalised per channel. IR is `[B, C_ir, H, W]` on the target grid (0.02 deg, 256 x 256 storm-centred),
MW is `[B, C_mw, h, w]` with `h = H / 8`. Every forward pass documents its shapes inline.

## Data sources for Milton (AL142024, 5 to 10 Oct 2024)

* GOES-16 ABI L2 CMIP (C08, C10, C13), Full Disk or CONUS: `s3://noaa-goes16/ABI-L2-CMIPF/2024/{279..284}/`
* ATMS L1B (SNPP, NOAA-20, NOAA-21): NASA GES DISC `SNPPATMSL1B` / `N20ATMSL1B` or NOAA CLASS SDR (TATMS/SATMS)
* ERA5 pressure-level temperature (labels), Copernicus CDS `reanalysis-era5-pressure-levels`
* IMERG Final/Late half-hourly `precipitationCal`, GES DISC `GPM_3IMERGHH`
* Best track: IBTrACS v04 (SID `2024280N19269`) or HURDAT2 (`AL142024`)

Training the prior and the proxy requires many storms: build scenes for every Atlantic and East
Pacific tropical cyclone 2017 to 2023 at 3-hourly cadence (ERA5 hours with an ABI scan within
7.5 min and an ATMS overpass within 90 min), keeping Milton fully held out.

```python
from vaughan.config import PipelineConfig
from vaughan.data.best_track import BestTrack
from vaughan.data.dataset import RawScenePaths, build_scene_cache, Normalizer, HurricaneSceneDataset

cfg = PipelineConfig()
bt = BestTrack.from_ibtracs_csv("ibtracs.NA.list.v04r01.csv", sid="2024280N19269")
lat, lon = bt.center_at(np.datetime64("2024-10-08T12:00"))
scene = RawScenePaths(time=np.datetime64("2024-10-08T12:00"), storm_lat=lat, storm_lon=lon,
                      goes_files={"C08": ..., "C10": ..., "C13": ...}, atms_file=..., era5_file=..., imerg_file=...)
paths = build_scene_cache([scene, ...], cfg.data, "artifacts/scenes")
```

## Training and inference

```python
norm = Normalizer.fit([xr.load_dataset(p) for p in train_paths]); norm.save(cfg.data.stats_path)
train_ds = HurricaneSceneDataset(train_paths, cfg.data, norm, augment=True)
rtm = AnalyticRTM(cfg.data.levels_hpa, cfg.data.ir_channels, cfg.data.mw_channels, cfg.data.grid.mw_downscale)

unet = train_unet(cfg, train_ds, val_ds, norm, rtm)          # stage 1: deterministic proxy
score, ema = train_score(cfg, train_ds)                       # stage 2: unconditional prior on ERA5/IMERG states
# optional stage 0: train_rtm_residual(cfg, train_ds, HybridRTM(rtm, NeuralRTMResidual(...)), norm)
```

```bash
python -m vaughan.inference.run_milton --scenes artifacts/scenes/MILTON_*.nc \
    --stats artifacts/norm_stats.json --unet artifacts/checkpoints/unet.pt \
    --score artifacts/checkpoints/score.pt --out artifacts/analysis --ensemble 8 --steps 500
```

The engine runs the U-Net once, then `ensemble_size` reverse chains of `n_steps` steps; each step
evaluates the score network, the Tweedie estimate, the RTM, and one backward pass. On a single
A100 at 256 x 256 the default (8 members x 500 steps) completes in well under a minute per scene,
against hours for a 4D-Var cycle.

## How the physics guidance works

`JointLikelihood.terms()` returns per-sample negative log-likelihood terms in sum form:

* U-Net proxy: `||x0 - x_det||^2 / (2 sigma_unet^2)` (the learned "background")
* IR and MW radiative constraints: `||TB_obs - H(x0)||^2 / (2 sigma^2)` over valid pixels, with `H` the
  differentiable RTM applied to the physical (de-normalised) state
* static stability: `relu(dT/dlnp - kappa T)^2`, which forbids super-adiabatic layers, plus a
  non-negativity penalty on rain rate

`GuidedScoreSampler._guided_score()` differentiates this through the Tweedie estimate and the score
network (DPS) and adds it to the prior score; the reverse SDE then weights both by `beta(t) dt`,
so `guidance_scale = 1` is the Bayesian posterior score. A per-step RMS cap and a soft clamp of the
Tweedie estimate keep early steps (where `x0_hat` is unreliable) on the prior manifold. The chain
stops at `final_denoise_t` and returns the Tweedie estimate so pixel-level diffusion noise does not
leak into precipitation through `expm1`.

## Verification included

`pytest -q vaughan/tests` (about 15 s on CPU) checks dataset shapes and unit ranges, RTM
differentiability and monotonic physics (more rain gives colder IR window and colder 183 GHz),
masked cross-attention with a sample lacking MW coverage, a full train-then-assimilate loop, and an
exact-prior DA test: with the closed-form `GaussianClimatologyScore` prior, enabling guidance cut
the IR misfit from about 11 K to 1.7 K, MW from 2.4 K to 1.2 K, drove the stability penalty to zero,
and improved both temperature and precipitation RMSE relative to the U-Net proxy alone.

## Honest caveats for production

* `AnalyticRTM` is a structural proxy. Fit `NeuralRTMResidual` on collocations, or swap in CRTM /
  RTTOV via their adjoints, and tighten `sigma_ir_K` / `sigma_mw_K` accordingly.
* ATMS footprints are elliptical and scan-angle dependent; `regrid_swath_to_target` treats them as
  points. A Backus-Gilbert or footprint-matching step improves the MW channel at the eyewall.
* ERA5 is a smooth 0.25 deg target; for sharper eyewall structure train labels on HWRF/HAFS
  analyses or aircraft dropsonde composites and treat ERA5 as a weak prior.
* Hydrostatic balance is applied as the static-stability consequence for T on p-levels; add
  geopotential to the state if you want balance enforced explicitly.
