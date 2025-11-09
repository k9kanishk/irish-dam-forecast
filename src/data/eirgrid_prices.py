# src/data/eirgrid_prices.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests


BASE_URL = "https://www.smartgriddashboard.com/DashboardService.svc/data"
CACHE_DIR = Path("data/raw/eirgrid_dam")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _date_str(d: datetime) -> str:
    # EirGrid uses e.g. "07-Nov-2025"
    return d.strftime("%d-%b-%Y")


def _fetch_eirgrid_json(date_from: datetime, date_to: datetime) -> dict:
    """
    Low-level call to EirGrid Smart Grid Dashboard.
    You *must* adjust the 'area' parameter to the actual one used for DAM prices
    (use your browser network tab to see the request when you view prices).
    """
    params = {
        "area": "dailymarketprices",   # TODO: set to real area name
        "region": "ALL",
        "datefrom": _date_str(date_from),
        "dateto": _date_str(date_to),
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _json_to_df(payload: dict) -> pd.DataFrame:
    """
    Convert EirGrid JSON payload to a DataFrame with columns:
        ts_utc (datetime64[ns, UTC]), dam_eur_mwh (float)
    You may need to tweak the key names once you see the real JSON.
    """
    rows = payload.get("Rows") or payload.get("rows") or payload
    if isinstance(rows, dict):
        rows = rows.get("Rows") or rows.get("rows")

    # ---- ADJUST THESE KEYS AFTER INSPECTING payload ----
    records = []
    for r in rows:
        # examples – update these once you see actual keys:
        ts = r.get("datetime") or r.get("dt") or r.get("DateTime")
        price = r.get("price") or r.get("val") or r.get("Value")
        if ts is None or price is None:
            continue
        ts = pd.to_datetime(ts, utc=True, errors="coerce")
        records.append({"ts_utc": ts, "dam_eur_mwh": float(price)})

    df = pd.DataFrame.from_records(records)
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    df = df.drop_duplicates("ts_utc", keep="last")
    df["dam_eur_mwh"] = pd.to_numeric(df["dam_eur_mwh"], errors="coerce")
    df = df.set_index("ts_utc").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def fetch_dam_recent(days: int = 21, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch last `days` of Irish DAM prices from EirGrid, with simple parquet caching.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    cache_fp = CACHE_DIR / f"eirgrid_dam_{start.date()}_{end.date()}.parquet"
    if cache_fp.exists() and not force_refresh:
        df = pd.read_parquet(cache_fp)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    payload = _fetch_eirgrid_json(start, end)
    df = _json_to_df(payload)

    df.to_parquet(cache_fp)
    return df
