from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterable
import datetime as dt
import xml.etree.ElementTree as ET

import pandas as pd
import requests

REPORT_NAME = "DAM 60Min Harmonised Reference Price"
GROUP = "Market Data"
LIST_URL = "https://reports.semopx.com/api/v1/documents/static-reports"
DOC_URL = "https://reports.semopx.com/documents"

CACHE_PQ = Path("data/raw/semopx_dam_60min_hrp.parquet")
RAW_DIR = Path("data/raw/semopx/hrp60_raw")


def _request_json(params: dict, retries: int = 4, timeout: int = 20) -> dict:
    last = None
    for a in range(1, retries + 1):
        try:
            r = requests.get(LIST_URL, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(min(6.0, 0.6 * (2 ** (a - 1))))
    raise RuntimeError(f"SEMO list API failed: {last}")


def _extract_items(payload: dict) -> List[dict]:
    for key in ("items", "Items", "data", "Data", "Reports", "reports"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    for v in payload.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def _download(resource_name: str) -> bytes:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fp = RAW_DIR / resource_name
    if fp.exists() and fp.stat().st_size > 0:
        return fp.read_bytes()

    url = f"{DOC_URL}/{resource_name}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    fp.write_bytes(r.content)
    return r.content


def _parse_hrp60_xml(raw: bytes) -> pd.DataFrame:
    """
    Parse SEMOpx HRP60 XML:
      Interval / StartTime / MarketPriceRoundedAmount
    """
    root = ET.fromstring(raw)

    def lname(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def val(el: ET.Element) -> str:
        if "v" in el.attrib:
            return str(el.attrib["v"]).strip()
        return (el.text or "").strip()

    rows = []
    # parse across all TimeSeries
    for ts_el in root.iter():
        if lname(ts_el.tag).lower() not in ("timeseries",):
            continue

        bidding_area = None
        for el in ts_el.iter():
            if lname(el.tag).lower() == "biddingarea":
                bidding_area = val(el)
                break

        for interval in ts_el.iter():
            if lname(interval.tag).lower() != "interval":
                continue

            start_txt = None
            price_txt = None
            for child in interval.iter():
                n = lname(child.tag).lower()
                if n == "starttime":
                    start_txt = val(child)
                elif n == "marketpriceroundedamount":
                    price_txt = val(child)

            if not start_txt or not price_txt:
                continue

            ts_utc = pd.to_datetime(start_txt, utc=True, errors="coerce")
            try:
                pr = float(price_txt)
            except Exception:
                pr = None

            if ts_utc is not pd.NaT and pr is not None:
                rows.append((ts_utc, pr, bidding_area))

    if not rows:
        raise RuntimeError("No Interval(StartTime, MarketPriceRoundedAmount) rows found.")

    df = pd.DataFrame(rows, columns=["ts_utc", "dam_60min_hrp_eur_mwh", "bidding_area"])
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["dam_60min_hrp_eur_mwh"] = pd.to_numeric(df["dam_60min_hrp_eur_mwh"], errors="coerce")
    df = df.dropna(subset=["ts_utc", "dam_60min_hrp_eur_mwh"]).sort_values("ts_utc")
    return df.reset_index(drop=True)


def _daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def update_hrp60_cache(
    days_back: int = 14,
    max_age_minutes: int = 15,
    bidding_area: Optional[str] = None,
) -> Path:
    """
    Update data/raw/semopx_dam_60min_hrp.parquet if stale.

    Pulls latest published reports for the last `days_back` days, merges with existing parquet, de-dups.
    """
    CACHE_PQ.parent.mkdir(parents=True, exist_ok=True)

    # Staleness check
    if CACHE_PQ.exists():
        age_min = (time.time() - CACHE_PQ.stat().st_mtime) / 60.0
        if age_min <= max_age_minutes:
            return CACHE_PQ

    # Existing cache
    if CACHE_PQ.exists():
        old = pd.read_parquet(CACHE_PQ)
        old["ts_utc"] = pd.to_datetime(old["ts_utc"], utc=True, errors="coerce")
    else:
        old = pd.DataFrame(columns=["ts_utc", "dam_60min_hrp_eur_mwh", "bidding_area", "trade_date"])

    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)

    new_parts = []
    for d in _daterange(start, end):
        d_str = d.isoformat()

        params = {
            "Group": GROUP,
            "ReportName": REPORT_NAME,
            "Date": d_str,
            "sort_by": "PublishTime",
            "order_by": "DESC",
            "page_size": 25,
            "page": 1,
        }
        payload = _request_json(params)
        items = _extract_items(payload)
        if not items:
            continue

        # pick newest published
        it = items[0]
        rn = it.get("ResourceName") or it.get("resourceName")
        if not rn:
            continue

        try:
            raw = _download(str(rn))
            df = _parse_hrp60_xml(raw)
            df["trade_date"] = d_str
            if bidding_area:
                df = df[df["bidding_area"].astype(str) == bidding_area]
            if not df.empty:
                new_parts.append(df)
        except Exception:
            continue

    if new_parts:
        new = pd.concat(new_parts, ignore_index=True)
        out = pd.concat([old, new], ignore_index=True)
        out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True, errors="coerce")
        out["dam_60min_hrp_eur_mwh"] = pd.to_numeric(out["dam_60min_hrp_eur_mwh"], errors="coerce")
        out = out.dropna(subset=["ts_utc", "dam_60min_hrp_eur_mwh"]).sort_values("ts_utc")

        # de-dupe (keep newest)
        out = out.drop_duplicates(["ts_utc", "bidding_area"], keep="last")
        out.to_parquet(CACHE_PQ, index=False)

    return CACHE_PQ


def load_hrp60(days: int = 21, bidding_area: Optional[str] = None) -> pd.DataFrame:
    """
    Load last `days` of HRP60 from cached parquet.
    """
    if not CACHE_PQ.exists():
        update_hrp60_cache(days_back=max(days, 14), max_age_minutes=0, bidding_area=bidding_area)

    df = pd.read_parquet(CACHE_PQ)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["dam_60min_hrp_eur_mwh"] = pd.to_numeric(df["dam_60min_hrp_eur_mwh"], errors="coerce")
    df = df.dropna(subset=["ts_utc", "dam_60min_hrp_eur_mwh"]).sort_values("ts_utc")

    if bidding_area:
        df = df[df["bidding_area"].astype(str) == bidding_area]

    cutoff = df["ts_utc"].max() - pd.Timedelta(days=days)
    return df[df["ts_utc"] >= cutoff].reset_index(drop=True)
