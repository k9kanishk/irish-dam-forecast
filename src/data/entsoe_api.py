# src/data/entsoe_api.py — resilient chunked fetch + optional SEMOpx fallback
from __future__ import annotations
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import pytz

from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# Optional SEMOpx fallback (only used if present)
try:
    from data.semopx_api import fetch_dam_hrp_recent as _fetch_semopx_recent
except Exception:
    _fetch_semopx_recent = None  # fallback not available

BZ = ZoneInfo("Europe/Brussels")  # ENTSO-E wants Brussels tz
AREA_CANDIDATES = (
    "IE",                    # Ireland
    "SEM",                   # SEM alias (sometimes works)
    "10YIE-1001A00010",      # EIC: Ireland
    "10Y1001A1001A59C",      # EIC: SEM
)

def _bz(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(BZ) if t.tzinfo is None else t.tz_convert(BZ)

def _query_da_prices(client: EntsoePandasClient, start, end) -> pd.Series:
    """Try multiple area codes; return UTC Series named 'dam_eur_mwh'."""
    last_err = None
    for area in AREA_CANDIDATES:
        try:
            # Some entsoe-py versions lack 'timeout'; try with then without
            try:
                s = client.query_day_ahead_prices(area, start=_bz(start), end=_bz(end), timeout=60)
            except TypeError:
                s = client.query_day_ahead_prices(area, start=_bz(start), end=_bz(end))
            if s is not None and len(s) > 0:
                s = s.tz_convert("UTC")
                s.name = "dam_eur_mwh"
                return s
        except Exception as e:
            last_err = e
            continue
    raise NoMatchingDataError(f"No DA price data for {start}..{end} in {AREA_CANDIDATES}: {last_err}")

def fetch_ie_dam_prices_entsoe(
    start_date: str,
    end_date: str,
    cache_dir: Path = Path("data/processed"),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch IE/SEM day-ahead prices for [start_date, end_date] (civil dates).
    Returns UTC-indexed DataFrame with 'dam_eur_mwh'.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"dam_ie_{start_date}_{end_date}.parquet"

    if fp.exists() and not force_refresh:
        df = pd.read_parquet(fp)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df["dam_eur_mwh"] = pd.to_numeric(df["dam_eur_mwh"], errors="coerce")
        return df

    token = os.getenv("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_TOKEN not set (env or .streamlit/secrets.toml).")

    client = EntsoePandasClient(api_key=token)
    # ENTSO-E end bound is exclusive, include entire end day
    ser = _query_da_prices(client, start_date, pd.Timestamp(end_date) + pd.Timedelta(days=1))
    df = ser.to_frame("dam_eur_mwh")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(fp)
    return df

def fetch_ie_dam_chunked(
    days: int = 21,
    chunk_days: int = 7,
    force_refresh: bool = False,
    delay_days: int = 3,
    fallback_semopx: bool = True,
) -> pd.DataFrame:
    """
    Fetch last `days` of prices using multiple small windows.
    - `delay_days` skips very recent days ENTSO-E might not have yet.
    - Catches NoMatchingDataError per chunk and continues.
    - If all chunks fail and SEMOpx fallback is enabled and available, uses it.
    """
    tz = pytz.timezone("Europe/Dublin")
    end_local = pd.Timestamp.now(tz).normalize() - pd.Timedelta(days=delay_days)
    start_local = end_local - pd.Timedelta(days=days)

    # Build chunk windows [start, end)
    windows = []
    cur = start_local
    while cur < end_local:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_local)
        windows.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt

    parts = []
    for s, e in windows:
        try:
            df_part = fetch_ie_dam_prices_entsoe(s, e, force_refresh=force_refresh)
            if df_part is not None and not df_part.empty:
                parts.append(df_part)
        except NoMatchingDataError:
            # Just skip this chunk; we'll try others
            continue
        except Exception:
            # Network or other error — skip this chunk too
            continue

    if not parts:
        if fallback_semopx and _fetch_semopx_recent is not None:
            # Use SEMOpx HRP as last resort for the same window
            try:
                sem_df = _fetch_semopx_recent(days=max(days, 14))
                if sem_df is not None and not sem_df.empty:
                    sem_df = sem_df.set_index("ts_utc")
                    if sem_df.index.tz is None:
                        sem_df.index = sem_df.index.tz_localize("UTC")
                    sem_df = sem_df.sort_index()
                    # conform: only dam_eur_mwh column with UTC index
                    sem_df = sem_df[["dam_eur_mwh"]]
                    return sem_df
            except Exception:
                pass
        # Nothing worked
        raise NoMatchingDataError("ENTSO-E returned no data for all chunks and SEMOpx fallback was unavailable/empty.")

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

def fetch_ie_dam_recent(days: int = 21, force_refresh: bool = True, delay_days: int = 3) -> pd.DataFrame:
    """Convenience wrapper around chunked fetch."""
    return fetch_ie_dam_chunked(days=days, chunk_days=7, force_refresh=force_refresh, delay_days=delay_days)

class Entsoe:
    """Lightweight helper for load/wind/solar with robust tz handling."""
    def __init__(self, token: str | None = None, area: str = "IE"):
        token = token or os.getenv("ENTSOE_TOKEN")
        if not token:
            raise RuntimeError("ENTSOE_TOKEN not set")
        self.client = EntsoePandasClient(api_key=token)
        self.area = area

    def _brussels(self, dt_like) -> pd.Timestamp:
        ts = pd.Timestamp(dt_like)
        return ts.tz_localize(BZ) if ts.tzinfo is None else ts.tz_convert(BZ)

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        try:
            s = _query_da_prices(self.client, start, end)
            return s
        except Exception:
            return pd.Series(dtype=float, name="dam_eur_mwh")

    def load_forecast(self, start: str, end: str) -> pd.Series:
        try:
            df = self.client.query_load_and_forecast(self.area, start=self._brussels(start), end=self._brussels(end))
            def _norm(c):
                try: return " ".join(map(str, c)).lower()
                except TypeError: return str(c).lower()
            cand = [c for c in df.columns if "forecast" in _norm(c)]
            col = cand[0] if cand else df.columns[-1]
            s = pd.to_numeric(df[col], errors="coerce"); s.name = "load_forecast_mw"
            return s
        except Exception:
            return pd.Series(dtype=float, name="load_forecast_mw")

    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        def _try(psr: str) -> pd.Series:
            try:
                s = self.client.query_wind_and_solar_forecast(self.area,
                        start=self._brussels(start), end=self._brussels(end), psr_type=psr)
                return s if s is not None else pd.Series(dtype=float)
            except (NoMatchingDataError, Exception):
                return pd.Series(dtype=float)
        solar    = _try("B16")
        wind_on  = _try("B19")
        wind_off = _try("B18")
        df = pd.concat({"solar_mw": solar, "wind_onshore_mw": wind_on, "wind_offshore_mw": wind_off}, axis=1).sort_index()
        cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df]
        if cols:
            df["wind_total_mw"] = df[cols].sum(axis=1, min_count=1)
        return df
