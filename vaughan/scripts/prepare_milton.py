"""
Build the Hurricane Milton scene cache from the public archives.

    python -m vaughan.scripts.prepare_milton --root data/milton \
        --start 2024-10-06T00 --end 2024-10-10T00 --step-hours 6 \
        [--times 2024-10-07T06 2024-10-07T18 ...] [--dry-run]

Any storm in IBTrACS works: --storm MELISSA --season 2025 --start 2025-10-25T00 --end 2025-10-29T00 --goes-satellite goes19
(GOES-East has been GOES-19 since April 2025; the Milton scenes came from GOES-16).

A live or very recent storm is not in IBTrACS yet and has no ERA5 for about five days. Use the NHC ATCF
b-deck for the track and build unlabelled scenes (--live), then rebuild with ERA5 when it exists:

    python -m vaughan.scripts.prepare_milton --root data/polo --track atcf --atcf-id EP172026 --storm POLO \
        --start 2026-09-21T00 --end 2026-09-25T00 --goes-satellite goes19 --live
    ...a week later:
    python -m vaughan.scripts.prepare_milton --root data/polo --track atcf --atcf-id EP172026 --storm POLO \
        --start 2026-09-21T00 --end 2026-09-25T00 --goes-satellite goes19 --rebuild

Steps: IBTrACS track -> analysis times and centres -> GOES-East (nearest scan), ATMS (nearest overpass
from SNPP/NOAA-20/NOAA-21), IMERG (containing half hour), ERA5 (one request per day) -> manifest.json
-> scene NetCDFs under <root>/scenes. Re-runs are incremental: existing files are not fetched again.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data.best_track import BestTrack
from ..data.dataset import build_scene_cache
from ..data.download import (assemble_manifest, download_atms, download_bdeck, download_era5, download_goes, download_ibtracs,
                             download_imerg, save_manifest, storm_area)

log = logging.getLogger("vaughan")


def analysis_times(start: str, end: str, step_hours: int, explicit) -> list:
    if explicit:
        return [np.datetime64(t) for t in explicit]
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{step_hours}h").to_numpy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--storm", default="MILTON", help="storm name (IBTrACS lookup key with --track ibtracs; label for the scene files)")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--basin", default="NA", help="IBTrACS basin list to download: NA (Atlantic) or EP (eastern North Pacific)")
    ap.add_argument("--track", default="ibtracs", choices=["ibtracs", "atcf"], help="track source: IBTrACS (final, lags weeks) or the NHC ATCF b-deck (live)")
    ap.add_argument("--atcf-id", default=None, help="ATCF storm id for --track atcf, e.g. EP172026 (Polo), AL142024 (Milton)")
    ap.add_argument("--live", action="store_true", help="live storm: skip ERA5, keep scenes without IMERG; scenes carry NaN labels and attrs labels")
    ap.add_argument("--no-era5", action="store_true", help="skip the ERA5 request (implied by --live)")
    ap.add_argument("--start", default="2024-10-06T00:00")
    ap.add_argument("--end", default="2024-10-10T00:00")
    ap.add_argument("--step-hours", type=int, default=6)
    ap.add_argument("--times", nargs="*", default=None, help="explicit analysis times (override start/end/step)")
    ap.add_argument("--goes-product", default=None, help="force ABI-L2-CMIPF or ABI-L2-CMIPC")
    ap.add_argument("--goes-satellite", default="goes16", choices=["goes16", "goes18", "goes19"],
                    help="GOES-East was GOES-16 until April 2025 and GOES-19 after; use goes19 for 2025 storms")
    ap.add_argument("--dry-run", action="store_true", help="resolve track and times, download nothing")
    ap.add_argument("--rebuild", action="store_true", help="rebuild scene NetCDFs even if they exist (raw files are reused)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = PipelineConfig()
    root = args.root
    raw = os.path.join(root, "raw")
    if args.track == "atcf":
        if not args.atcf_id:
            ap.error("--track atcf needs --atcf-id, e.g. EP172026")
        bd = download_bdeck(os.path.join(raw, "atcf"), args.atcf_id)
        bt = BestTrack.from_atcf_bdeck(bd)
        if args.storm == "MILTON" and bt.name:
            args.storm = bt.name
        args.season = int(args.atcf_id[-4:])
    else:
        ib = download_ibtracs(os.path.join(raw, "ibtracs"), basin=args.basin)
        bt = BestTrack.from_ibtracs_csv(ib, name=args.storm, season=args.season)
    times = analysis_times(args.start, args.end, args.step_hours, args.times)
    times = [t for t in times if bt.times[0] <= np.datetime64(t, "ns") <= bt.times[-1]]
    if not times:
        log.error(f"no analysis times inside the track ({bt.times[0]} to {bt.times[-1]})")
        return 1
    centers = [bt.center_at(t) for t in times]
    log.info(f"{args.storm} {args.season} SID {bt.sid}: {len(times)} analysis times from {times[0]} to {times[-1]}")
    for t, (la, lo) in zip(times, centers):
        extra = ""
        if hasattr(bt, "wind_kt"):
            k = int(np.argmin(np.abs(bt.times - np.datetime64(t, "ns"))))
            extra = f"  best track {bt.wind_kt[k]:.0f} kt" + (f" {bt.mslp_hpa[k]:.0f} hPa" if hasattr(bt, "mslp_hpa") else "")
        log.info(f"  {np.datetime_as_string(t, unit='m')}  centre {la:6.2f}N {lo:7.2f}E{extra}")
    if args.dry_run:
        return 0
    live = args.live
    skip_era5 = args.no_era5 or live

    goes = download_goes(times, centers, cfg.data.ir_channels, os.path.join(raw, args.goes_satellite), satellite=args.goes_satellite,
                         tol_min=cfg.data.ir_time_tolerance_min, product=args.goes_product)
    atms = download_atms(times, centers, os.path.join(raw, "atms"), tol_min=cfg.data.mw_time_tolerance_min)
    imerg = download_imerg(times, os.path.join(raw, "imerg"))
    area = storm_area([c[0] for c in centers], [c[1] for c in centers])
    era5 = {} if skip_era5 else download_era5(times, cfg.data.levels_hpa, area, os.path.join(raw, "era5"), tag=f"{args.storm}{args.season}")
    if skip_era5:
        log.info("ERA5 skipped: scenes will carry NaN temperature labels (attrs labels); rebuild with --rebuild once ERA5T covers the period")

    scenes = assemble_manifest(times, centers, goes, atms, era5, imerg, storm_name=args.storm, require_labels=not live)
    manifest = os.path.join(root, "manifest.json")
    save_manifest(scenes, manifest)
    n_mw = sum(1 for s in scenes if s.atms_files)
    n_unl = sum(1 for s in scenes if s.era5_file is None or s.imerg_file is None)
    log.info(f"manifest: {len(scenes)} scenes ({n_mw} with an ATMS overpass, {n_unl} without full labels) -> {manifest}")
    paths = build_scene_cache(scenes, cfg.data, os.path.join(root, "scenes"), overwrite=args.rebuild)
    log.info(f"scene cache: {len(paths)} files under {os.path.join(root, 'scenes')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
