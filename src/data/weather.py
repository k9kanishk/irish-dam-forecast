# src/data/weather.py
from __future__ import annotations
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = "Europe/Dublin"

def _today_ie():
    return datetime.now(ZoneInfo(TZ)).date()

def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Returns hourly weather indexed by local Dublin time (tz-naive).
    Uses the forecast endpoint for short spans (<=16d), ERA5 archive otherwise.
    We request data in UTC to avoid DST ambiguity, then convert.
    """
    start_d = pd.to_datetime(start).date()
    end_d   = pd.to_datetime(end).date()
    today   = _today_ie()

    # the archive can’t go beyond yesterday
    if (end_d - start_d).days + 1 > 16:
        end_d = min(end_d, today - timedelta(days=1))
        if end_d < start_d:  # user asked only future → pull last 15 days
            start_d = end_d - timedelta(days=15)

    short_span = (end_d - start_d).days + 1 <= 16
    base = "https://api.open-meteo.com/v1/forecast" if short_span \
        else "https://archive-api.open-meteo.com/v1/era5"

    # variable names differ slightly between endpoints
    hourly_vars = ["wind_speed_100m", "temperature_2m", "cloud_cover"] if short_span \
        else ["windspeed_100m", "temperature_2m", "cloudcover"]

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "timezone":   "UTC",                    # <<< request in UTC
        "start_date": str(start_d),
        "end_date":   str(end_d),
        "hourly":     ",".join(hourly_vars),
    }

    def _get(p):
        r = requests.get(base, params=p, timeout=60)
        r.raise_for_status()
        return r.json()["hourly"]

    try:
        h = _get(params)
    except requests.HTTPError:
        # last resort: shrink to <=16 days
        end_try = end_d
        start_try = max(start_d, end_try - timedelta(days=15))
        params.update(start_date=str(start_try), end_date=str(end_try))
        h = _get(params)

    # normalize keys
    wind = h.get("wind_speed_100m", h.get("windspeed_100m"))
    cloud = h.get("cloud_cover", h.get("cloudcover"))

    # build UTC index, convert to Dublin, then drop tz → tz-naive local time
    ts_utc = pd.to_datetime(h["time"]).tz_localize("UTC")
    idx = ts_utc.tz_convert(TZ).tz_localize(None)

    df = pd.DataFrame(
        {
            "wind100m_ms":   wind,
            "temperature_2m": h["temperature_2m"],
            "cloud_cover":    cloud,
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df
