# Vaughan scientific and engineering review

Reviewed 20 September 2026 UTC (19 September local time).

## Overall assessment

Vaughan is a credible, working research prototype for storm-centred satellite retrieval. Its strongest demonstrated component is the deterministic cross-attention U-Net. The saved evidence does not yet establish calibrated posterior uncertainty, a useful accuracy advantage from diffusion, retrieval of independently verified fine-scale hurricane structure, or improved forecasts.

The work is worth continuing. The most valuable next investment is in observation physics, experimental controls, uncertainty calibration and reproducibility. Another architecture variation is a lower priority.

The defensible present claim is: **a reproducible satellite-to-reanalysis retrieval baseline, with an experimental differentiable physics-guided refinement and documented failure modes on two hurricane case studies.** Claims of replacing operational assimilation, supplying a calibrated Bayesian posterior, or supplying observationally resolved eyewall structure exceed the evidence.

## What I inspected and verified

I reviewed the package's ingestion, normalization, architectures, SDE, guidance, analytic and scattering operators, training, inference and tests; the experiment/evaluation scripts; dashboard source and numerical inputs; saved Melissa and WeatherNext results; root and artifact checkpoints; the technical report's scientific narrative; and relevant portions of the supplied Cannon et al. precipitation-retrieval paper. Interview documents and old patch archives are supporting material, not primary validation evidence; I did not exhaustively compare every historical archive or visually review every document page.

Independent checks performed:

- Parsed all 41 Python files under `vaughan` and `dashboard` successfully.
- Ran the package suite on CPU: 20 tests passed initially, and one failed to set up because Windows denied access to pytest's temporary directory. That remaining test passed when rerun with an explicit review temporary directory. All 21 test cases therefore passed across the two runs.
- Ran tests from an isolated working directory because the training tests write checkpoints to relative `artifacts/checkpoints` paths.
- Loaded the four supplied checkpoints with `weights_only=True` and inspected their steps and metadata.
- Loaded all 17 Melissa cached scenes: their temperature and precipitation fields were finite, and their microwave masks and timing metadata were inspected.
- Ran the root U-Net checkpoint on all 17 Melissa scenes using the root normalizer. Reproduced the saved core RMSE values to a maximum difference of **0.0000011 K**. This is strong evidence that the supplied U-Net artifacts genuinely reproduce the stored results.
- Independently recomputed the saved CSV summaries and executed minimal reproductions of five numerical/engineering issues below.

I did not retrain the real models, run all 8 x 500 diffusion chains, recreate the missing multi-season training archive, benchmark a GPU, rerun WeatherNext, or run the dashboard's browser suite. Historical training and sampler results remain supplied evidence rather than freshly reproduced runs. The CPU dependencies were installed separately under `.review-deps`, with a local `.review-cache`; project requirements and implementation were not edited.

Reproduction scripts: `review/check_findings.py`, `review/verify_real_proxy.py`. Machine-readable findings: `review/check_results.json`, `review/real_proxy_results.json`.

## What the numerical evidence actually says

The following are means of the per-scene RMSE values, not pooled pixel RMSE. Temperature errors are against ERA5 at 300 hPa.

| Melissa subset | Scenes | Analysis core RMSE, K | U-Net core RMSE, K | Mean core spread, K |
|---|---:|---:|---:|---:|
| All | 17 | 1.15042 | 1.14965 | 0.09236 |
| ATMS present | 8 | 0.90561 | 0.91299 | 0.08925 |
| ATMS absent | 9 | 1.36803 | 1.36002 | 0.09512 |

Across all scenes, mean domain RMSE is 0.92653 K for the analysis and 0.93624 K for the U-Net: a 0.00972 K reduction, approximately 1%. Domain error improves on 16/17 scenes, but core error improves on only 7/17. Pooling equally sized core boxes gives RMSE of 1.36010 K for the analysis and 1.35559 K for the U-Net, again slightly favouring the proxy.

My fresh proxy run also beat the fixed training-mean temperature profile: mean core RMSE was 1.14965 K versus 4.18337 K. This is a useful positive result. It is a weak climatology baseline, however; persistence, a storm-relative climatology, infrared-only models, and conventional retrievals would be stronger comparators. This check does not validate the unseen training split or independent observational accuracy.

The roughly 12.5-fold ratio between mean core error and mean core spread is a serious warning. It is not a formal calibration statistic: the CSV stores spatial mean standard deviation, whereas a spread-skill comparison needs matching second moments and independent verification uncertainty. Nevertheless, the size of the mismatch makes a claim of calibrated uncertainty untenable without substantial further evidence.

Retrieved warm-core amplitude averages 3.2958 K with ATMS and 2.4500 K without it. The alternating pattern is scientifically interesting, but natural overpass versus non-overpass comparisons confound time, storm evolution and observation availability. Run paired observation denial at the same timestamps, using identical random seeds, and separately test a model trained for missing microwave data.

For Milton, the dashboard raw scalar inputs give mean core RMSE of 1.7506 K for analysis, 1.7681 K for the U-Net, and 2.1619 K for physics-only microwave. These rounded dashboard numbers should be distinguished from newer evaluation tables and full-resolution output metrics. They support modest refinement, not a large diffusion benefit.

## Scientific assessment

### 1. The Bayesian interpretation needs correction

For a noisy diffusion state, the exact posterior score is

`grad log p_t(x_t | y) = grad log p_t(x_t) + grad log p_t(y | x_t)`.

The likelihood on the right integrates over possible clean states. Evaluating the observation likelihood at the Tweedie mean is an approximation to that integral. The original [DPS paper](https://arxiv.org/abs/2209.14687) explicitly describes approximate posterior sampling.

Vaughan adds further approximations: a tanh transform of the clean estimate, guidance disabled above a selected time, displacement clipping, scalar uncertainty inflation, adaptive Langevin steps and a final denoised point estimate. These may improve numerical behaviour, but `guidance_scale=1` does not make this an exact Bayesian sampler.

In particular, `sigma_t^2 / alpha_t^2` is not generally the conditional covariance of the clean state. For a unit Gaussian prior with a variance-preserving forward process, the exact conditional variance is `sigma_t^2`, not that ratio. For a general prior it depends on the distribution and observation state. Propagating uncertainty through an observation operator also requires something like `H' C H'^T`, not one mean temperature variance for every channel and pixel.

The proxy `x_det` is already a function of the same IR and MW observations used in the radiance terms. Multiplying an independent Gaussian proxy penalty and an observation likelihood therefore risks counting the same information twice. It can be defined as a useful conditional energy model, but it is not automatically the posterior for an independent forecast background. A conditional prior or explicit model of joint residual dependence would make the interpretation clearer.

The all-sky weights depend on the current state but are computed under `no_grad`, and the cost omits the Gaussian log-determinant term for state-dependent covariance. This is closer to an iteratively reweighted objective than the gradient of the stated heteroscedastic probability density. The symmetric-cloud concept has a legitimate basis in [Geer and Bauer's work](https://www.ecmwf.int/en/elibrary/74560-enhanced-use-all-sky-microwave-observations-sensitive-water-vapour-cloud-and), but using that concept does not by itself establish the claimed posterior.

Recommended decisive test: a linear observation model with a known multivariate Gaussian prior, comparing sampled posterior means **and covariances** to the analytical answer across observation errors, masks and step counts. The existing exact-prior test only establishes improved misfit, not correct posterior moments.

### 2. The RTM is a structural surrogate; its derivatives remain unvalidated

The per-channel audit is one of the best parts of the project. Finding tens-of-K operator biases before trusting their gradients is good scientific practice. However, a constant correction changes the intercept, not the temperature or hydrometeor Jacobian. Low residual scatter does not establish that a channel responds correctly to a warm-core perturbation.

The analytic operator lacks humidity, skin temperature, surface emissivity, surface pressure, liquid cloud and a complete upper atmosphere. Fixed weighting functions cannot adjust to the actual column. In contrast, [RTTOV's documented inputs](https://www.nwpsaf.eu/site/software/rttov/) include temperature, water vapour, surface parameters and viewing geometry, with optional hydrometeors and other constituents.

In opaque cloud, the IR implementation makes all three channels approach the same sampled cloud-top temperature. This is a strong simplification of spectrally different channels. Its prescribed cloud top can reach 150 hPa although the temperature state stops at 200 hPa, so temperatures above the state top cannot actually be retrieved by this operator.

ATMS sounding channels have approximately 32 km nadir footprints in the 50-90 GHz range, rather than the 16-18 km suggested by one grid comment; see [NOAA's instrument description](https://www.star.nesdis.noaa.gov/cris/ATMS_background.php). A regridded pixel is not a new independent footprint. Interpolation and footprint overlap induce spatially correlated errors, which the diagonal likelihood currently ignores.

ScatteringRTM is a useful extension, but its clear-sky gas opacity is constructed to reproduce the same approximate weighting functions. Its ice optics assume soft spheres, a prescribed size distribution and fixed optical-property temperature. Matching the analytic clear-sky result is an internal consistency check, not external RTM validation. Validate brightness temperatures and Jacobians against a reference solver across clear, cloudy, icy, coastal and scan-angle regimes before restoring channels to the likelihood.

When ice is supplied to AnalyticRTM, both the IR cloud term and the microwave depression become independent of surface rain. ScatteringRTM likewise has no explicit surface-rain dependence in that mode. A fresh derivative check confirmed the analytic outputs had no dependency on a trainable rain input. Rain may still be inferred indirectly through correlations in the learned prior and proxy, but it is no longer directly constrained by a radiative rain Jacobian. The state design and claims should make this explicit.

### 3. Fine output spacing does not establish fine-scale information

The executed small preset uses 128 x 128 fields on the original domain, approximately 4.4 km north-south spacing, not the README's nominal 256 x 256 grid. [ERA5 has approximately 31 km native resolution](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5). Interpolating ERA5 to either grid does not create independent eye-scale temperature truth.

With ten temperature levels and surface rain at each pixel, the inverse problem has a large null space. Once the IR likelihood is dropped and only five microwave sounding channels are retained, most fine structure necessarily comes from learned correlations and regularization. Sharp-looking fields alone cannot distinguish real resolution recovery from plausible interpolation or hallucination.

Use independent dropsondes, aircraft/radar observations, or carefully characterized higher-resolution analyses, with matched spatial support. Report averaging kernels or local Jacobian singular values, and estimate which vertical and horizontal modes are actually observed. Do not describe a superior fit to the assimilated radiances as independent validation.

IMERG is itself a merged satellite retrieval using microwave and IR information; it is not an independent rain truth for this task. Its recommended interpretation is an average over the half-hour period, according to [NASA's documentation](https://gpm.nasa.gov/data/imerg). Validate extreme-rain and accumulation skill against appropriately matched radar/gauges, with their uncertainties documented.

### 4. Thermodynamic constraints are narrower than physical balance

The dry static-stability penalty has a sensible sign and is useful for catching grossly unstable profiles. It does not enforce hydrostatic balance independently, moist stability, mass conservation, momentum balance or temporal consistency. The thickness calculation follows diagnostically from temperature and uses dry temperature rather than virtual temperature.

The state includes 1000, 925 and 850 hPa without surface pressure or a below-ground mask. Where surface pressure is lower than a pressure level, that level lies below the physical surface; this matters both in a deep cyclone and over terrain. Interpolated/extrapolated reanalysis values there should not be interpreted as atmospheric retrievals or included indiscriminately in physics losses.

The nonnegativity penalty is identically zero after `state_to_physical` already clamps rain to nonnegative values. The clamp also gives a zero gradient in the negative latent-rain region. This is a numerical design issue, not evidence that the penalty has enforced rain physics.

### 5. The current experiment is retrospective

The matching window accepts observations before and after analysis time. In the actual Melissa caches, six of the eight microwave scenes use later overpasses, at offsets **+6 to +57 minutes**. Storm centres come from retrospective best track. Neither is wrong for a retrospective analysis, but both matter for claims of real-time availability.

The U-Net is passed zenith angle but not observation age, and the likelihood has no temporal evolution operator or motion correction. A 90-minute mismatch during rapid intensification can represent a different storm state. Evaluate past-only observations, operational centre estimates, age-dependent errors and a sequence-based model. Compute end-to-end latency including observation availability and preprocessing, not just GPU runtime.

### 6. Generalization and forecast claims need stronger experiments

Holding out a season is a good choice. However, the training archive is not present locally, so its 805/252 scene composition, number of independent storms, preprocessing history and full exclusion rules could not be verified. The phrase '805 storms' in the dashboard is inconsistent with the documented 805 **scenes**.

The extensive Milton debugging makes Milton a development case even if it was excluded from gradient training. Melissa is more informative as a second case, but two related hurricane case studies are not broad validation. The common architecture train/validation gap is consistent with limited data or domain shift; it does not prove that architecture is irrelevant. One seed per variant is insufficient to rank small changes reliably.

Use several held-out storms and seasons, plus weakening systems, nonstorms, missing sensors, coastlines and transfer between satellites. Bootstrap by storm or long time blocks, not by pixels. Report core and environmental errors separately, conditional bias, rain thresholds, radial profiles and uncertainty coverage. Include simple proxy ensembles or quantile models as probabilistic baselines.

The WeatherNext experiment is a separate HRES-initialized forecast. The local metadata explicitly identifies its HRES input; there is no Vaughan-to-WeatherNext initialization shown. At +18 hours the saved ensemble mean is 965.9 hPa versus observed 908 hPa, and 91.8 kt versus 150 kt. This documents a forecast difficulty, not evidence that Vaughan would fix it. A paired forecast experiment initialized with and without a balanced Vaughan increment is required. The public [WeatherNext repository](https://github.com/google-deepmind/weathernext) provides context for the external model, not validation of the coupling.

## Engineering findings, ranked

| Priority | Finding and evidence | Consequence / corrective action |
|---|---|---|
| High | `_coarsen_sample` averages the normalized log-rain state (`data/dataset.py:421`). For physical rain [0,9;0,9], I reproduced 2.1623 mm/h rather than 4.5 mm/h. | It changes the target from area-mean rain to a geometric-type mean. Coarsen physical rain/ice before transforming; version the changed labels and rerun affected metrics. |
| High | `--no-rtm --allsky` is not an observation-free ablation (`inference/run_milton.py:184`, `assimilation/guidance.py:120`). I reproduced an intended 1,000,000 K error becoming 40 K. | Use explicit disabled-term flags or exactly zero weights. Test flag combinations. `--scatter-fit` can also restore MW weights after `--no-rtm`. |
| High | `calibrate_scatter` runs before `calibrate_from_audit` (`inference/run_milton.py:192-200`). | Audit calibration can overwrite fitted channel bias/error weights. Establish precedence, reject incompatible settings, and save the final effective calibration. |
| High | Training tests save to the ordinary `artifacts/checkpoints` path (`tests/test_smoke.py`). | Running the advertised test command from the project root can overwrite real checkpoints. Every test must use `tmp_path` and an isolated checkpoint directory. Existing artifact weights were already tiny smoke checkpoints before this review. |
| Medium | `to_dataset` uses unbiased `std` for one member (`inference/run_milton.py:111`). | I reproduced all-NaN temperature and rain spreads. Match the sampler's explicit one-member behaviour or clearly mark spread unavailable. |
| Medium | Ice export inverse-transforms the normalized ensemble mean (`inference/run_milton.py:113`). | This is not the arithmetic physical ensemble mean. A two-member example gives 2.1623 rather than 4.5. Transform members first, then calculate mean/spread, as already done for rain. |
| Medium | Three ice-column discretizations disagree (`dataset.integrate_ice`, `rtm.py:130`, `scatter.py:213`). | Constant q=1e-4 over the state levels gives 0.8158, 0.8923 and 0.8668 kg/m2 respectively. Use common layer boundaries and integration throughout; include the above-200-hPa domain consistently. |
| Medium | Training loaders use `drop_last=True` inside `while step < total`. | With a positive `max_steps` and fewer scenes than batch size, the loader is empty and the loop cannot progress. Fail early or allow a smaller final batch. |
| Medium | Existing normalizer files are silently reused; checkpoint config restoration is partial. | Changed splits/preprocessing can silently use stale statistics; old root weights contain no config metadata. Store full config, scene/split hashes, code revision, normalization hash, optimizer/scheduler/scaler/RNG state and calibration together. |
| Medium | Warm-core environment radius and centre averaging are specified in pixels (`physics/constraints.py:52`). | Changing downscale changes the physical diagnostic. Define radii and core support in km and use the same definition everywhere. |
| Medium | Masks are not enforced throughout the convolutional MW context path. The all-missing attention fallback allows every token, whose learned features need not be zero. | Current tests prove finite outputs, not invariance to masked values. Add invariance tests, mask inputs in the model and handle the all-missing branch explicitly. |
| Medium | Training target MSE has no validity mask; normalizer reductions are not NaN-aware. | Future scenes with missing temperature labels can poison statistics and training. Distinguish missing precipitation from zero rain and validate every cached scene before admission. |

Additional numerical caution: the corrector guidance cap is scaled using predictor `beta*dt`, although the corrector uses its own adaptive step. Thus it does not guarantee the documented bound on corrector displacement. Avoid synchronized Python scalar extraction on every GPU step where practical, and use per-member random generators instead of mutating global RNG state.

Exports would also benefit from validity masks, posterior members or sufficient verification products, calibration metadata, software/model identifiers, and explicitly labelled raw versus bias-corrected simulated radiances. The existing output computes `H(mean state)`, which generally differs from the ensemble mean of `H(member)`. Full CF compliance has not been demonstrated by a conformance check.

## Reproducibility and presentation

The root checkpoints are real small-preset models: U-Net step 6,000 with about 9.56M state-dict elements, score step 60,000 with about 26.97M. The `artifacts/checkpoints` versions are step 40/60 debug networks with about 0.64M/0.59M elements. The expected `artifacts/norm_stats.json` is absent. A manifest identifying the authoritative model bundle would prevent accidental misuse.

The dashboard's architecture table reports 2.89 K / 4.72 mm/h validation for the baseline, while its machine-readable training metadata and technical report retain 2.55 K / 4.05 mm/h. This could reflect different evaluation aggregation or runs, but it needs explicit provenance. The local Melissa CSV and figure lack the additional ERA5 warm-core series discussed in the later dashboard text; the Colab M6 code explains how to add them, but those regenerated result files are not supplied here.

README commands and claims have drifted: the example scene constructor uses the old singular `atms_file` argument; root duplicate scripts use obsolete package paths; runtime and test-count descriptions vary between documents. The README's full-grid A100 sub-minute claim is not supported by a benchmark in the inspected artifacts and should not be equated with the reported small-grid T4 timing.

The dashboard engineering is sensible: small static modules, content-hashed assets, a compact quantized field buffer, lazy decoding, accessibility controls and browser tests. But a polished page cannot substitute for a machine-readable experiment ledger. Generate its numerical tables and captions from one versioned result source and retain raw, unrounded validation products.

## Recommended development sequence

1. **Establish a trustworthy baseline.** Isolate test outputs; fix rain downscaling, ablation precedence and export issues; save the authoritative model/config bundle; regenerate a single consistent set of tables. Version changed data rather than mixing old checkpoints and new labels.
2. **Test whether uncertainty is meaningful.** Add analytical Gaussian posterior checks, coverage/rank/CRPS diagnostics, spread-skill checks on matching supports, and multi-seed proxy ensembles. Separate observation, representation, model and sampling uncertainty.
3. **Validate the observation operator.** Build matched collocations; compare forward radiances and finite-difference/autograd Jacobians to a reference RTM. Handle humidity, surface/upper-atmosphere state, footprints and geometry. Freeze calibration before the final storm tests.
4. **Run paired, multi-storm experiments.** Compare direct proxy, infrared-only proxy, proxy plus deterministic correction, diffusion without radiances, and full guidance under identical observation availability and seeds. Include a past-only operating mode and independent aircraft/radar verification.
5. **Attempt forecast coupling only after retrieval validation.** Construct balanced increments for a forecast model's full required state, run paired forecasts and test track/intensity/rain impacts. Faster sampling or distillation should follow proof of incremental value.

The strongest research opportunity is to determine when trustworthy radiative guidance adds information beyond a learned proxy, and whether that benefit survives missing sensors, storm evolution and independent verification. The folder already contains much of the machinery needed to answer that question; its current evidence is strongest as a baseline and a diagnosis of why the harder claims remain unproven.
