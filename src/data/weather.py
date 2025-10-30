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
    Returns hourly weather with index in Europe/Dublin (tz-naive).
    Chooses Open-Meteo forecast API for short spans (<=16 days) and ERA5 archive otherwise.
    """
    start_d = pd.to_datetime(start).date()
    end_d   = pd.to_datetime(end).date()
    today   = _today_ie()

    # never request past 'today' (forecast can go future a bit, archive cannot)
    if end_d > today:
        end_d = today

    span_days = (end_d - start_d).days + 1

    if span_days <= 16:
        # near-term forecast API (supports future)
        base = "https://api.open-meteo.com/v1/forecast"
        hourly_vars = ["wind_speed_100m", "temperature_2m", "cloud_cover"]
    else:
        # ERA5 archive (historical only) – clamp to yesterday
        end_d = min(end_d, today - timedelta(days=1))
        if end_d < start_d:
            # if user asked entirely in the future, pull last 15 days instead
            start_d = end_d - timedelta(days=15)
        base = "https://archive-api.open-meteo.com/v1/era5"
        hourly_vars = ["windspeed_100m", "temperature_2m", "cloudcover"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": TZ,
        "start_date": str(start_d),
        "end_date": str(end_d),
        "hourly": ",".join(hourly_vars),
    }

    def _request(p):
        r = requests.get(base, params=p, timeout=60)
        r.raise_for_status()
        return r.json()["hourly"]

    try:
        h = _request(params)
    except requests.HTTPError:
        # One more try: shrink to last 16 days
        end_try = end_d
        start_try = max(start_d, end_try - timedelta(days=15))
        params.update(start_date=str(start_try), end_date=str(end_try))
        h = _request(params)

    # normalize keys between endpoints
    ts = pd.to_datetime(h["time"])
    if "wind_speed_100m" in h:
        wind = h["wind_speed_100m"]
    else:
        wind = h["windspeed_100m"]
    if "cloud_cover" in h:
        cloud = h["cloud_cover"]
    else:
        cloud = h["cloudcover"]

    idx = (ts
    .tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
    .tz_localize(None))

    df = pd.DataFrame(
    {"wind100m_ms": h["windspeed_100m"],
        "temperature_2m": h["temperature_2m"],
        "cloud_cover": cloud,},index=idx,)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df
