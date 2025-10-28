from __future__ import annotations
import pandas as pd
import requests

def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Hourly weather for [start, end] inclusive. Switches forecast/ERA5 automatically."""
    start_d = pd.to_datetime(start).date()
    end_d   = pd.to_datetime(end).date()
    days = (end_d - start_d).days + 1

    base = (
        "https://api.open-meteo.com/v1/forecast"               # <= 16 days
        if days <= 16
        else "https://archive-api.open-meteo.com/v1/era5"      # > 16 days
    )

    params = {
        "latitude": lat,
        "longitude": lon,
        # correct var names for v1
        "hourly": ["windspeed_100m", "temperature_2m", "cloudcover"],
        "start_date": str(start_d),
        "end_date": str(end_d),
        "timezone": "Europe/Dublin",
    }

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]

    dt = pd.to_datetime(h["time"])
    df = pd.DataFrame(
        {
            "wind100m_ms": h["windspeed_100m"],
            "temp2m_c":    h["temperature_2m"],
            "cloudcover":  h["cloudcover"],
        },
        index=dt,
    )
    return df
