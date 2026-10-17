# Ian M1 verification — 17 October 2026

Implemented: standard-library public HWM fetcher with completeness check, point-asset
proximity baseline, evidence envelope, synthetic asset demonstration, pilot specification
and offline contract tests. This is a retrospective review precursor, not flood mapping.

## Verified

- Live fetcher completed against the hosted Ian layer: 82 features and independent count
  query 82; event IDs all 325; bounded WGS84 point geometry validated. Source bytes SHA-256:
  `d285d3524102ffb81c74f01f3f629c76e889e625d46330c34b8c082067f6fe91`.
  See `IAN_SOURCE_MANIFEST.json` for query and retrieval time. Raw snapshot remains in
  ignored `data/ian/`; subsequent source edits can change the hash.
- Synthetic demonstration: one nearby review candidate (HWM 43887, distance rounds to
  0 m), one unknown with no nearby evidence (nearest mark 19,448.01 m), one outside pilot.
  Outputs are in ignored `artifacts/ian-screen.json`, reproducible via the pilot commands.
  These three cases are not accuracy or customer validation.
- Five new tests passed: status separation, metre/radius behavior, wrong event, duplicate
  identifiers, invalid coordinates/CRS, partial response rejection and count mismatch.
- Regression check: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, then
  `python -m pytest -q vaughan/tests -k 'not test_cli' --basetemp artifacts/pytest-noncli`:
  **38 passed, 4 deselected**, 38.76 s (includes the five new tests).
- New modules compile; `git diff --check` clean. No diff to existing hurricane data/model/
  physics/assimilation/training/inference code, Colab experiments, results, dashboard or
  scientific report relative to starting revision `865590d`.

## Verification limits

The initial full suite hit temporary-directory permission errors (20 passed, 21 setup
errors). A rerun with an explicit temporary directory progressed but did not complete the
existing hurricane CLI subprocess checks; attempts with thread limits also did not finish
those checks during this session and were interrupted. Four existing `test_cli*` checks
therefore remain unverified; do not report a full-suite pass. No matching leftover Vaughan
CLI test processes were found in the cleanup check.

No customer assets/inspection outcomes or satellite flood scenes were available locally.
The source copy has not been reconciled with authoritative STN records. Publication timing
is unknown and field flag dates are later than the proposed decision deadline. No flood
accuracy, inspection benefit, delivery SLA, shoreline change, depth, damage or recovery
result is established. M2/M3 gates remain open. Velo's routine water context stays separate.

The README backlink to Lillie US is retained. The hub's runtime/documentation pins are
unchanged: a local unpublished milestone is not a coordinated dependency release. Existing
hurricane evaluation caveats, including proxy agreement and persistence smoothing limits,
remain intact. Work is confined to Vaughan on `codex/ian-inspection-pilot`.
