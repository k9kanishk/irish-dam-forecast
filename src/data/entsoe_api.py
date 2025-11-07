# src/data/entsoe_api.py — robust version
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytz

from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# ENTSO-E expects Brussels timezone for queries
BZ = ZoneInfo("Europe/Brussels")

# Try these area identifiers in order until one returns data
AREA_CANDIDATES = (
    "IE",                    # Ireland
    "SEM",                   # Single Electricity Market (fallback some installs use)
    "10YIE-1001A00010",      # EIC: Ireland
    "10Y1001A1001A59C",      # EIC: SEM (legacy mappings)
)


def _bz(ts) -> pd.Timestamp:
    """Make a Brussels-tz aware timestamp for ENTSO-E queries."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(BZ)
    return t.tz_convert(BZ)


def _query_da_prices(client: EntsoePandasClient, start, end) -> pd.Series:
    """
    Try multiple area codes; return a UTC-tz Series named 'dam_eur_mwh'.
    Handles entsoe-py versions that don't accept 'timeout' kwarg.
    """
    last_err = None
    for area in AREA_CANDIDATES:
        try:
            # Try with 'timeout'; fall back to calling without if unsupported
            try:
                s = client.query_day_ahead_prices(area, start=_bz(start), end=_bz(end), timeout=60)
            except TypeError:
                s = client.query_day_ahead_prices(area, start=_bz(start), end=_bz(end))
            if s is not None and len(s) > 0:
                s = s.tz_convert("UTC")
                s.name = "dam_eur_mwh"
                return s
        except NoMatchingDataError as e:
            last_err = e
            continue
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
    Fetch IE/SEM day-ahead prices from ENTSO-E for [start_date, end_date] (civil dates).
    Returns a UTC-indexed DataFrame with one column 'dam_eur_mwh'.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"dam_ie_{start_date}_{end_date}.parquet"

    if fp.exists() and not force_refresh:
        df = pd.read_parquet(fp)
        # Normalize cache formats
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df["dam_eur_mwh"] = pd.to_numeric(df["dam_eur_mwh"], errors="coerce")
        return df

    token = os.getenv("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_TOKEN not set (add to env or .streamlit/secrets.toml).")

    client = EntsoePandasClient(api_key=token)

    # ENTSO-E end bound is exclusive → include the full end day
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
) -> pd.DataFrame:
    """
    Fetch the last `days` of prices using multiple small windows for speed/reliability.
    Uses a `delay_days` offset (default 3) to avoid very recent days that ENTSO-E may not publish yet.
    Returns a UTC-indexed DataFrame with 'dam_eur_mwh'.
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
        df_part = fetch_ie_dam_prices_entsoe(s, e, force_refresh=force_refresh)
        if df_part is not None and not df_part.empty:
            parts.append(df_part)

    if not parts:
        raise NoMatchingDataError("ENTSO-E returned no data for the requested window.")

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def fetch_ie_dam_recent(days: int = 21, force_refresh: bool = True, delay_days: int = 3) -> pd.DataFrame:
    """
    Return last `days` of prices up to (today - delay_days). Uses chunked fetch.
    """
    return fetch_ie_dam_chunked(days=days, chunk_days=7, force_refresh=force_refresh, delay_days=delay_days)


class Entsoe:
    """Small helper for other ENTSO-E fundamentals."""
    def __init__(self, token: str | None = None, area: str = "IE"):
        token = token or os.getenv("ENTSOE_TOKEN")
        if not token:
            raise RuntimeError("ENTSOE_TOKEN not set")
        # Avoid passing 'timeout' here — some entsoe-py versions don't accept it
        self.client = EntsoePandasClient(api_key=token)
        self.area = area

    def _brussels(self, dt_like) -> pd.Timestamp:
        ts = pd.Timestamp(dt_like)
        if ts.tzinfo is None:
            return ts.tz_localize(BZ)
        return ts.tz_convert(BZ)

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        """
        Convenience wrapper using the same robust area/tz logic.
        """
        try:
            s = _query_da_prices(self.client, start, end)
            return s
        except Exception:
            return pd.Series(dtype=float, name="dam_eur_mwh")

    def load_forecast(self, start: str, end: str) -> pd.Series:
        """Fetch load forecast (Series), tolerant to column naming changes."""
        try:
            df = self.client.query_load_and_forecast(
                self.area, start=self._brussels(start), end=self._brussels(end)
            )
            def _norm(c):
                try:
                    return " ".join(map(str, c)).lower()
                except TypeError:
                    return str(c).lower()
            cand = [c for c in df.columns if "forecast" in _norm(c)]
            col = cand[0] if cand else df.columns[-1]
            s = pd.to_numeric(df[col], errors="coerce")
            s.name = "load_forecast_mw"
            return s
        except Exception:
            return pd.Series(dtype=float, name="load_forecast_mw")

    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        """Fetch wind/solar forecast; missing series become empty."""
        def _try(psr: str) -> pd.Series:
            try:
                s = self.client.query_wind_and_solar_forecast(
                    self.area, start=self._brussels(start), end=self._brussels(end), psr_type=psr
                )
                return s if s is not None else pd.Series(dtype=float)
            except (NoMatchingDataError, Exception):
                return pd.Series(dtype=float)

        solar    = _try("B16")  # Solar
        wind_on  = _try("B19")  # Wind Onshore
        wind_off = _try("B18")  # Wind Offshore

        df = pd.concat(
            {"solar_mw": solar, "wind_onshore_mw": wind_on, "wind_offshore_mw": wind_off},
            axis=1
        ).sort_index()

        cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df]
        if cols:
            df["wind_total_mw"] = df[cols].sum(axis=1, min_count=1)
        return df
