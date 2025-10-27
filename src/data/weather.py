from __future__ import annotations
import requests
import pandas as pd

BASE = 'https://api.open-meteo.com/v1/forecast'


def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': ','.join(['wind_speed_100m','temperature_2m','cloud_cover']),
        'start_date': start,
        'end_date': end,
        'timezone': 'Europe/Dublin'
    }
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()['hourly']
    dt = pd.to_datetime(js['time'])
    df = pd.DataFrame({'wind100m_ms': js['wind_speed_100m'],
                       't2m_c': js['temperature_2m'],
                       'cloud_pct': js['cloud_cover']}, index=dt)
    df.index.name = 'ts'
    return df
