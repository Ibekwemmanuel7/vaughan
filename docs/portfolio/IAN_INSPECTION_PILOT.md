# Ian inspection pilot v0.1

Status: retrospective design plus an implemented point-evidence baseline. No customer,
asset inventory, independent inspection labels, flood mask or operational service is yet secured.
Part of [Lillie US](https://github.com/Ibekwemmanuel7/Lillie_US).

## Customer decision and geography

Proposed buyer: a Lee County commercial-property asset manager, working through an
environmental consultant. Decision: which sites should enter the first staffed post-event
inspection queue, subject to emergency authority access restrictions. A review candidate
is not a dispatch order or a finding of structural damage. Buyer access, staffing capacity
and the usefulness of a 72-hour delivery deadline require discovery; no outreach is claimed.

Freeze the first study to Hurricane Ian, 28–30 September 2022, and WGS84 bounding box
[-82.3, 26.4, -81.7, 26.8] (west, south, east, north): coastal Lee County/Fort Myers,
Sanibel and adjacent inland controls. This rectangle is not the county boundary. Restrict
customer assets to the rectangle; use asset footprints and entrance/access points in the
later pilot. The implemented baseline accepts points only. It cannot clear roads or sites.

## Data, provenance and separation

- Available now: [public Ian HWM layer](https://services.arcgis.com/3wFbqsFPLeKqOlIK/ArcGIS/rest/services/High_Water_Marks_Ian/FeatureServer/0),
  named `HWMs_Ian_USGS`, event ID 325. The bounded query returned 82 point records.
  This is a hosted USGS-labelled copy; direct STN endpoints were not successfully resolved
  in this session. Quality codes are preserved, not assigned undocumented meanings.
  Snapshot quality-code counts are 1:44, 2:27, 3:9, 5:2. Flag dates span 3–10 October
  2022, after the proposed 72-hour cutoff; this is later retrospective evidence.
  Do not interpret `stillwater` as a depth or assume a vertical datum from a code.
- Authoritative reference acquisition planned: [USGS FEV download portal](https://apps.usgs.gov/fev/data-download-portal),
  survey quality definitions, measured elevations/datums and storm-tide time series.
  [USGS describes HWM collection](https://www.usgs.gov/news/featured-story/usgs-determines-extent-hurricane-induced-flooding-which-can-help-communities).
  HWMs document local peak evidence; they are neither a negative sample nor a simultaneous extent map.
- Planned prediction inputs: dated pre/post Sentinel-1 acquisitions, Sentinel-2/Landsat clear
  observations, USGS 3DEP terrain and pre-event permanent-water context. Audit actual scene
  coverage, rights, processing level, orbit and timestamps before selecting imagery. No such
  imagery has been downloaded or processed in this milestone.
- Independent outcome labels: permissioned customer inspections with asset ID, coordinates,
  visit time, photographs, inundation evidence, access obstruction, reviewer and unresolved
  status. Use surveyed HWMs for development/context only, never score this proximity method
  against the same marks. Separate held-out inspection sites, including dry controls.
- The example CSV has synthetic points, including one deliberately located at a source mark.
  It is a software demonstration, not a customer inventory or accuracy sample. Keep customer
  geometry and raw datasets outside source commits and tenant-isolated in any later service.

## Timing and baselines

Operational proposal: freeze a replay cutoff at 72 hours after 2022-09-28T19:05Z
(2022-10-01T19:05Z), then evaluate 7-, 14- and 30-day recovery updates separately.
Record acquisition interval, first upstream availability, ingest time, processing time and
review time independently. Unknown availability makes an input ineligible for a causal
72-hour performance claim. HWM flag dates are field-recording dates, not publication or
peak-water times. The current snapshot is retrospective only and does not satisfy this cutoff.

Pre-event plan: inspect same-orbit observations from 30–7 days before landfall and the
nearest suitable pre-event scene; document antecedent rainfall/tide differences. Post-event:
first suitable observation inside 72 hours, then days 7/14/30 with actual timestamps and
coverage, rather than interpolated claims about the peak. Match water-level references
within one hour where available; otherwise report temporal mismatch and abstain from
depth validation. For shoreline comparisons require co-registration and comparable tide,
and report resolution/tide uncertainty. Shoreline processing remains planned.

Frozen baseline B0: current inspect-all policy, measured at the same daily inspection
capacity (20 sites/day proposed, customer confirmation required). B1: public event HWM
proximity, default 100 m spherical distance, sensitivity at 50/100/250 m. This is an
unvalidated review radius, not a physical flood buffer or a probability. B2, planned:
simple pre/post radar change with pre-event permanent-water exclusion. Compare any new
method with B0/B1/B2 on identical eligible sites, information cutoffs and review budgets.
Break ties by criticality fixed before scoring and then stable asset ID; never by outcome labels.

## Uncertainty and acceptance

Point evidence is spatially sparse and targeted. Urban radar shadow, vegetation, wind-roughened
water, terrain, geolocation, standing permanent water, tides and observation lag can hide or
mimic flooding. No-nearby-HWM means unknown. Exposure is not interior water depth,
damage, safe access or recovery. The baseline retains raw quality codes and null availability;
it does not emit calibrated confidence. Broad extent/depth inference requires other methods.

Milestone M1 software gate: reproducible bounded query and hash, matching source count,
event and CRS validation, rejection of duplicate IDs/invalid coordinates/partial responses,
and distinct nearby/unknown/empty/outside states. Run offline contract tests and a real
snapshot demonstration. Passing M1 establishes software behavior only.

Proposed M2 scientific gate, frozen before scoring: 200 independently inspected sites,
including at least 50 flooded and 50 dry controls and at least 30 each urban/vegetated sites
(strata may overlap). Split development and test by spatial blocks of at least 1 km; no
repeated asset across splits. Register selection probabilities, nonresponse and adjudication.
Report weighted precision, recall, false positives, omissions and unresolved fraction with
spatial-block bootstrap 95% intervals; omit accuracy claims if sampling cannot support them.
Gate: recall lower 95% bound at least 0.90, precision lower bound at least 0.60, unresolved
at most 20%, and more independently confirmed flooded sites among the first 20 inspections
than B0 and B1, with paired 95% interval for improvement above zero. Thresholds are proposed
decision policy, not demonstrated performance. Insufficient cases/labels means insufficient
evidence, not pass. Keep all methods' unknowns in the denominator and publish best/worst
case sensitivity; never silently discard difficult sites. Have two reviewers adjudicate
conflicts while blinded to method scores. The HWM baseline may fail these gates.

Proposed M3 delivery gate: a prospective event and buyer exercise, at least 30 scheduled
runs including outage/missing-source cases; 95% processed within two hours of eligible
input arrival and reviewed within four staffed hours. Report upstream and end-to-end
p50/p95 separately, including unknown timing and missed deadlines. All unknown/no-data
sites retain review visibility. No automated safe-site downgrade. Until M2/M3 pass, keep
human review and research-only status. No operational dispatch commitment is made.

## Ownership and next boundary

Vaughan owns this event evidence workflow. Velo supplies routine water-resource, watershed
and water-quality context when available; this code adds no Velo dependency or routine
water-monitoring service. Inundation, shoreline disturbance, asset polygons, access routes
and recovery classification remain planned. Preserve the hurricane pipeline, results and
evaluation history; its scores do not validate this pilot.

## Reproduce M1

From the repository root, Python 3.10+ (stdlib for the baseline; pytest for tests):

```powershell
python -m vaughan.flood.fetch_ian
python -m vaughan.flood.inspection --evidence data/ian/hwms.geojson --assets examples/flood/demo_assets.csv --out artifacts/ian-screen.json
python -m pytest -q vaughan/tests/test_flood_inspection.py vaughan/tests/test_flood_fetch.py
```

Network access is needed only for fetching. Upstream edits can change count/hash; freeze
and retain each raw snapshot outside Git. Commit a provenance summary, not customer data.
