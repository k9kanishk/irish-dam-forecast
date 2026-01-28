#!/usr/bin/env python3
"""
Fetch SEMOpx "DAM 60Min Harmonised Reference Price" via the official SEMOpx Report API.

Why this instead of clicking the website?
- The Market Results page is a UI; the underlying data is provided via the SEMOpx Website Report API.
- This approach is faster, deterministic, and less likely to break than Selenium/Playwright.

Sources:
- SEMOpx Website Report API PDF (2-step: list -> download) and no auth required.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, List

import pandas as pd
import requests
import xml.etree.ElementTree as ET


REPORT_NAME = "DAM 60Min Harmonised Reference Price"
GROUP = "Market Data"
LIST_URL = "https://reports.semopx.com/api/v1/documents/static-reports"
DOC_URL = "https://reports.semopx.com/documents"  # /{ResourceName}

DEFAULT_OUT_CSV = Path("data/raw/semopx_dam_60min_hrp.csv")
DEFAULT_OUT_PARQUET = Path("data/raw/semopx_dam_60min_hrp.parquet")
CACHE_DIR = Path("data/raw/semopx/hrp60_raw")


@dataclass
class ReportItem:
    resource_name: str
    publish_time: str
    date: str  # trade date


def _request_json(url: str, params: dict, retries: int = 5, timeout: int = 30) -> dict:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            # exponential-ish backoff
            time.sleep(min(8.0, 0.7 * (2 ** (attempt - 1))))
    raise RuntimeError(f"Failed request after {retries} retries: {last_err}")


def _extract_items(payload: dict) -> List[dict]:
    """
    API responses vary slightly by version; handle common shapes.
    We expect a header + list of items.
    """
    # Common patterns
    for key in ("items", "Items", "data", "Data", "reports", "Reports"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]

    # Sometimes payload is {"header": {...}, "data": [..]}
    for key in ("data", "Data"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]

    # Worst case: payload itself is a list-ish under unknown key
    # Try: find first list value in dict
    for v in payload.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v

    raise RuntimeError(f"Unexpected JSON shape; keys={list(payload.keys())[:20]}")


def _find_best_report_for_date(date_yyyy_mm_dd: str, page_size: int = 50) -> Optional[ReportItem]:
    """
    Query the report list and select the newest published file for the given Trade Date.
    """
    params = {
        "Group": GROUP,
        "ReportName": REPORT_NAME,
        "Date": date_yyyy_mm_dd,           # exact date
        "sort_by": "PublishTime",
        "order_by": "DESC",
        "page_size": page_size,
        "page": 1,
    }
    payload = _request_json(LIST_URL, params=params)
    items = _extract_items(payload)

    # Normalize fields defensively
    norm = []
    for it in items:
        rn = it.get("ResourceName") or it.get("resourceName") or it.get("resourcename")
        pt = it.get("PublishTime") or it.get("publishTime") or it.get("publishtime")
        d  = it.get("Date") or it.get("date")
        if rn and pt and d and str(d).startswith(date_yyyy_mm_dd):
            norm.append(ReportItem(resource_name=str(rn), publish_time=str(pt), date=str(d)))

    if not norm:
        return None

    # Already sorted by PublishTime DESC, but keep safe:
    norm.sort(key=lambda x: x.publish_time, reverse=True)
    return norm[0]


def _download_resource(resource_name: str, cache_dir: Path = CACHE_DIR) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / resource_name

    if fp.exists() and fp.stat().st_size > 0:
        return fp.read_bytes()

    url = f"{DOC_URL}/{resource_name}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    fp.write_bytes(r.content)
    return r.content


def _localname(tag: str) -> str:
    # Strip namespace: "{ns}Tag" -> "Tag"
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_hrp60_xml(raw: bytes) -> pd.DataFrame:
    """
    Parse typical ENTSO-E/SDAC-style XML (Publication_MarketDocument-like).
    We extract (start + (position-1)*resolution) and price.amount.

    Output columns:
      ts_utc (datetime64[ns, UTC]), dam_60min_hrp_eur_mwh (float)
    """
    root = ET.fromstring(raw)

    # Find all TimeSeries blocks (some files contain multiple series)
    ts_blocks = [el for el in root.iter() if _localname(el.tag).lower() == "timeseries"]
    if not ts_blocks:
        raise RuntimeError("No TimeSeries found in XML; format may have changed.")

    frames = []
    for ts in ts_blocks:
        # Find Period
        period = None
        for el in ts.iter():
            if _localname(el.tag).lower() == "period":
                period = el
                break
        if period is None:
            continue

        # Find period start
        start_txt = None
        resolution_txt = None
        for el in period.iter():
            ln = _localname(el.tag).lower()
            if ln == "start":
                start_txt = (el.text or "").strip()
            if ln == "resolution":
                resolution_txt = (el.text or "").strip()

        if not start_txt:
            continue

        # Default resolution PT60M if not present
        # Parse ISO duration like PT60M / PT1H
        step = pd.Timedelta(hours=1)
        if resolution_txt:
            if resolution_txt.upper() in ("PT60M", "PT1H"):
                step = pd.Timedelta(hours=1)
            elif resolution_txt.upper() == "PT30M":
                step = pd.Timedelta(minutes=30)
            else:
                # fallback: assume hourly
                step = pd.Timedelta(hours=1)

        start = pd.to_datetime(start_txt, utc=True, errors="coerce")
        if pd.isna(start):
            continue

        rows = []
        for point in [el for el in period.iter() if _localname(el.tag).lower() == "point"]:
            pos = None
            price = None
            for child in point.iter():
                ln = _localname(child.tag).lower()
                txt = (child.text or "").strip()

                if ln == "position":
                    try:
                        pos = int(txt)
                    except Exception:
                        pass

                # Most common: "price.amount"
                if "price" in ln and ("amount" in ln or ln == "price"):
                    try:
                        price = float(txt)
                    except Exception:
                        pass

            if pos is None or price is None:
                continue

            ts_utc = start + (pos - 1) * step
            rows.append((ts_utc, price))

        if rows:
            tmp = pd.DataFrame(rows, columns=["ts_utc", "dam_60min_hrp_eur_mwh"])
            frames.append(tmp)

    if not frames:
        raise RuntimeError("Parsed XML but found no (timestamp, price) points.")

    out = pd.concat(frames, ignore_index=True)
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
    out["dam_60min_hrp_eur_mwh"] = pd.to_numeric(out["dam_60min_hrp_eur_mwh"], errors="coerce")
    out = out.dropna().sort_values("ts_utc")

    # If multiple series overlap, keep last (newest)
    out = out.drop_duplicates("ts_utc", keep="last").reset_index(drop=True)
    return out


def _parse_csv(raw: bytes) -> pd.DataFrame:
    # robust-ish CSV parsing
    s = raw.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(s))
    # Try to infer columns
    cols = {str(c).strip().lower(): c for c in df.columns}
    # candidate timestamp columns
    tcol = cols.get("ts_utc") or cols.get("timestamp") or cols.get("time") or list(df.columns)[0]
    pcol = cols.get("eur/mwh") or cols.get("price") or cols.get("dam_eur_mwh") or list(df.columns)[-1]

    out = pd.DataFrame({
        "ts_utc": pd.to_datetime(df[tcol], utc=True, errors="coerce"),
        "dam_60min_hrp_eur_mwh": pd.to_numeric(df[pcol], errors="coerce"),
    }).dropna().sort_values("ts_utc")

    out = out.drop_duplicates("ts_utc", keep="last").reset_index(drop=True)
    return out


def _daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out_csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--out_parquet", default=str(DEFAULT_OUT_PARQUET))
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    all_days = []
    for d in _daterange(start, end):
        d_str = d.isoformat()
        item = _find_best_report_for_date(d_str)
        if item is None:
            print(f"[WARN] No HRP60 report found for {d_str}", file=sys.stderr)
            continue

        raw = _download_resource(item.resource_name)

        # Parse based on extension
        rn = item.resource_name.lower()
        try:
            if rn.endswith(".xml"):
                df = _parse_hrp60_xml(raw)
            elif rn.endswith(".csv"):
                df = _parse_csv(raw)
            else:
                # try XML first, then CSV
                try:
                    df = _parse_hrp60_xml(raw)
                except Exception:
                    df = _parse_csv(raw)
        except Exception as e:
            print(f"[WARN] Failed parsing {item.resource_name} for {d_str}: {e}", file=sys.stderr)
            continue

        df["trade_date"] = d_str
        all_days.append(df)

        print(f"[OK] {d_str}: {len(df)} rows from {item.resource_name} (published {item.publish_time})")

    if not all_days:
        print("No data downloaded/parsed for requested window.", file=sys.stderr)
        return 2

    out = pd.concat(all_days, ignore_index=True).sort_values("ts_utc")
    out = out.drop_duplicates("ts_utc", keep="last").reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    out_pq = Path(args.out_parquet)
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_pq, index=False)

    print(f"Saved: {out_csv} ({len(out):,} rows)")
    print(f"Saved: {out_pq}  ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
