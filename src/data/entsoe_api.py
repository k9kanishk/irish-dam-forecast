# src/data/entsoe_api.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import pytz
from zoneinfo import ZoneInfo
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from concurrent.futures import ThreadPoolExecutor, as_completed

TZ_QUERY = ZoneInfo("Europe/Brussels")

def fetch_ie_dam_chunked(days: int = 21, chunk_days: int = 7, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch last `days` in chunks (7d by default). Faster + fewer timeouts.
    """
    tz = pytz.timezone("Europe/Dublin")
    end_local = pd.Timestamp.now(tz).normalize()
    start_local = end_local - pd.Timedelta(days=days)

    # Build chunk windows
    windows = []
    cur = start_local
    while cur < end_local:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_local)
        windows.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt

    parts = []
    # sequential is safest; if you want parallel, set max_workers=2
    for s, e in windows:
        parts.append(fetch_ie_dam_prices_entsoe(s, e, force_refresh=force_refresh))

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def fetch_ie_dam_prices_entsoe(
    start_date: str,
    end_date: str,
    cache_dir: Path = Path("data/processed"),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch IE day-ahead prices from ENTSO-E and return a UTC-indexed df with 'dam_eur_mwh'."""
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
        raise RuntimeError("ENTSOE_TOKEN not set.")

    client = EntsoePandasClient(api_key=token)
    tz = pytz.timezone("Europe/Dublin")
    start = tz.localize(pd.Timestamp(start_date))
    end   = tz.localize(pd.Timestamp(end_date)) + pd.Timedelta(days=1)  # end exclusive

    ser = client.query_day_ahead_prices("IE", start=start, end=end)

    if ser is None or len(ser) == 0:
        raise RuntimeError("ENTSO-E returned empty series for IE day-ahead prices.")

    ser = ser.tz_convert("UTC")
    df = ser.to_frame("dam_eur_mwh")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(fp)
    return df


def fetch_ie_dam_recent(days: int = 21, force_refresh: bool = True) -> pd.DataFrame:
    """Return last `days` of IE prices (UTC-indexed df)."""
    tz = pytz.timezone("Europe/Dublin")
    today = pd.Timestamp.now(tz).normalize()
    start = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    return fetch_ie_dam_prices_entsoe(start, end, force_refresh=force_refresh)


class Entsoe:
    """Small wrapper to fetch other fundamentals consistently."""
    def __init__(self, token: str | None = None, area: str = "IE"):
        token = token or os.getenv("ENTSOE_TOKEN")
        if not token:
            raise RuntimeError("ENTSOE_TOKEN not set")
        self.client = EntsoePandasClient(api_key=token)  # keep constructor simple
        self.area = area

    def _brussels(self, dt_like) -> pd.Timestamp:
        ts = pd.Timestamp(dt_like)
        if ts.tzinfo is None:
            return ts.tz_localize(TZ_QUERY)
        return ts.tz_convert(TZ_QUERY)

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        try:
            s = self.client.query_day_ahead_prices(self.area,
                                                   start=self._brussels(start),
                                                   end=self._brussels(end))
            if s is None or len(s) == 0:
                return pd.Series(dtype=float, name="dam_eur_mwh")
            s.name = "dam_eur_mwh"
            return s
        except Exception:
            return pd.Series(dtype=float, name="dam_eur_mwh")

    def load_forecast(self, start: str, end: str) -> pd.Series:
        df = self.client.query_load_and_forecast(self.area,
                                                 start=self._brussels(start),
                                                 end=self._brussels(end))
        def _norm(c):
            try: return " ".join(map(str, c)).lower()
            except TypeError: return str(c).lower()
        cand = [c for c in df.columns if "forecast" in _norm(c)]
        col = cand[0] if cand else df.columns[-1]
        s = pd.to_numeric(df[col], errors="coerce")
        s.name = "load_forecast_mw"
        return s

    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        def _try(psr: str) -> pd.Series:
            try:
                s = self.client.query_wind_and_solar_forecast(self.area,
                          start=self._brussels(start), end=self._brussels(end),
                          psr_type=psr)  # B16=Solar, B19=Wind On, B18=Wind Off
                return s if s is not None else pd.Series(dtype=float)
            except NoMatchingDataError:
                return pd.Series(dtype=float)

        solar   = _try("B16")
        wind_on = _try("B19")
        wind_off= _try("B18")

        df = pd.concat(
            {"solar_mw": solar, "wind_onshore_mw": wind_on, "wind_offshore_mw": wind_off},
            axis=1
        ).sort_index()

        cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df]
        if cols:
            df["wind_total_mw"] = df[cols].sum(axis=1, min_count=1)
        return df
