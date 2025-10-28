# src/data/weather.py
from __future__ import annotations
import pandas as pd
import requests

def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Return hourly weather for [start, end] (inclusive) with columns:
      - wind100m_ms
      - temp2m_c
      - cloudcover
    Uses forecast API for <=16 days, ERA5 archive otherwise.
    """
    start_d = pd.to_datetime(start).date()
    end_d   = pd.to_datetime(end).date()
    days = (end_d - start_d).days + 1

    # Choose endpoint based on window length
    if days <= 16:
        base = "https://api.open-meteo.com/v1/forecast"
    else:
        base = "https://archive-api.open-meteo.com/v1/era5"

    params = {
        "latitude": lat,
        "longitude": lon,
        # Correct variable names for Open-Meteo v1
        "hourly": ["windspeed_100m", "temperature_2m", "cloudcover"],
        "start_date": str(start_d),
        "end_date": str(end_d),
        "timezone": "Europe/Dublin",
    }

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()["hourly"]

    # Build DataFrame
    dt = pd.to_datetime(js["time"])
    df = pd.DataFrame(
        {
            "wind100m_ms": js.get("windspeed_100m"),
            "temp2m_c": js["temperature_2m"],
            "cloudcover": js["cloudcover"],
        },
        index=dt,
    )

    return df
