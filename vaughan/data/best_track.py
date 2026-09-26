"""
Storm-centre interpolation from best-track data (IBTrACS CSV or NHC HURDAT2) for storm-centred
domains. Milton (AL142024) is available in IBTrACS v04r01 (SID 2024279N21265) and HURDAT2.

    bt = BestTrack.from_ibtracs_csv("ibtracs.NA.list.v04r01.csv", name="MILTON", season=2024)
    lat, lon = bt.center_at(np.datetime64("2024-10-08T12:00"))
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _read_ibtracs(path: str) -> pd.DataFrame:
    """IBTrACS CSV: row 1 holds units; basin code "NA" (North Atlantic) must not be parsed as NaN."""
    return pd.read_csv(path, low_memory=False, skiprows=[1], keep_default_na=False, na_values=[""])


@dataclass
class BestTrack:
    times: np.ndarray   # datetime64[ns], sorted
    lats: np.ndarray
    lons: np.ndarray    # degrees east, -180..180

    @classmethod
    def from_ibtracs_csv(cls, path: str, sid: Optional[str] = None, name: Optional[str] = None, season: Optional[int] = None) -> "BestTrack":
        """Select a storm by IBTrACS SID, or by NAME + SEASON (e.g. name="MILTON", season=2024)."""
        df = _read_ibtracs(path)
        if sid is not None:
            df = df[df["SID"] == sid]
        else:
            df = df[(df["NAME"].str.upper() == name.upper()) & (df["SEASON"].astype(int) == int(season))]
            if df["SID"].nunique() != 1:
                raise ValueError(f"{name} {season}: found SIDs {sorted(df['SID'].unique())}; pass sid= explicitly")
        t = pd.to_datetime(df["ISO_TIME"]).values
        bt = cls(t, df["LAT"].astype(float).values, df["LON"].astype(float).values)
        bt.wind_kt = pd.to_numeric(df["USA_WIND"], errors="coerce").values
        bt.sid = str(df["SID"].iloc[0])
        return bt

    @staticmethod
    def list_storms(path: str, seasons, basins=("NA",), min_wind_kt: float = 64.0) -> pd.DataFrame:
        """Storms (one row each) in the given seasons/basins that reached at least min_wind_kt (USA_WIND).
        Useful for assembling the training archive."""
        df = _read_ibtracs(path)
        df = df[df["SEASON"].astype(int).isin(list(seasons)) & df["BASIN"].isin(list(basins))]
        df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")
        g = df.groupby("SID").agg(name=("NAME", "first"), season=("SEASON", "first"), max_wind=("USA_WIND", "max"),
                                  start=("ISO_TIME", "min"), end=("ISO_TIME", "max"))
        return g[g["max_wind"] >= min_wind_kt].reset_index()

    @classmethod
    def from_atcf_bdeck(cls, path: str) -> "BestTrack":
        """NHC/JTWC ATCF "b-deck" best track (https://ftp.nhc.noaa.gov/atcf/btk/bep172026.dat).

        The b-deck is the only best track that exists while a storm is active: it is updated with every
        advisory, whereas IBTrACS lags by days to weeks. Format (comma separated, one line per fix and
        wind radius): BASIN, CY, YYYYMMDDHH, TECHNUM, TECH, TAU, LatN/S (tenths), LonE/W (tenths), VMAX (kt),
        MSLP (hPa), TY, RAD, ... A fix with 34, 50 and 64 kt radii appears three times; one row per time
        is kept (the first). Also fills wind_kt, mslp_hpa, status (DB, TD, TS, HU, ...) and name."""
        times, lats, lons, winds, mslps, status, names = [], [], [], [], [], [], []
        seen = set()
        with open(path) as f:
            for line in f:
                p = [x.strip() for x in line.split(",")]
                if len(p) < 11 or p[4] != "BEST":
                    continue
                t = np.datetime64(f"{p[2][:4]}-{p[2][4:6]}-{p[2][6:8]}T{p[2][8:10]}:00")
                if t in seen:
                    continue
                seen.add(t)
                lat = float(p[6][:-1]) / 10.0 * (1 if p[6].endswith("N") else -1)
                lon = float(p[7][:-1]) / 10.0 * (1 if p[7].endswith("E") else -1)
                times.append(t), lats.append(lat), lons.append(lon)
                winds.append(float(p[8]) if p[8] else np.nan), mslps.append(float(p[9]) if p[9] else np.nan)
                status.append(p[10]), names.append(p[27] if len(p) > 27 else "")
        if not times:
            raise ValueError(f"{path}: no BEST fixes found")
        order = np.argsort(np.array(times, dtype="datetime64[ns]"))
        bt = cls(np.array(times, dtype="datetime64[ns]")[order], np.array(lats)[order], np.array(lons)[order])
        bt.wind_kt = np.array(winds)[order]
        bt.mslp_hpa = np.array(mslps)[order]
        bt.status = [status[i] for i in order]
        bt.name = next((n for n in reversed([names[i] for i in order]) if n and n not in ("INVEST",)), "")
        bt.sid = os.path.basename(path).split(".")[0].upper().lstrip("B")
        return bt

    @classmethod
    def from_hurdat2(cls, path: str, storm_id: str = "AL142024") -> "BestTrack":
        times, lats, lons = [], [], []
        with open(path) as f:
            active = False
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if parts[0].startswith("AL") or parts[0].startswith("EP"):
                    active = parts[0] == storm_id
                    continue
                if active and len(parts) > 6:
                    times.append(np.datetime64(f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}T{parts[1][:2]}:{parts[1][2:]}"))
                    lat = float(parts[4][:-1]) * (1 if parts[4].endswith("N") else -1)
                    lon = float(parts[5][:-1]) * (1 if parts[5].endswith("E") else -1)
                    lats.append(lat), lons.append(lon)
        return cls(np.array(times, dtype="datetime64[ns]"), np.array(lats), np.array(lons))

    def center_at(self, t: np.datetime64) -> Tuple[float, float]:
        """Linear interpolation of the centre position (degrees) at time t."""
        x = (self.times - self.times[0]) / np.timedelta64(1, "s")
        xt = (np.datetime64(t, "ns") - self.times[0]) / np.timedelta64(1, "s")
        return float(np.interp(xt, x, self.lats)), float(np.interp(xt, x, self.lons))
