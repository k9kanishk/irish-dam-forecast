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


def _parse_hrp60_xml(raw: bytes, debug: bool = False) -> pd.DataFrame:
    """
    Parse SEMOpx DAM_60MinHarmonisedReferencePrice XML.

    Observed tags (from your debug):
      - TimeSeries (2)
      - Interval (48 total, likely 24 per series)
      - StartTime (timestamp)
      - MarketPriceRoundedAmount (EUR/MWh)

    Returns columns:
      ts_utc (UTC), dam_60min_hrp_eur_mwh (float), bidding_area (str|None)
    """
    root = ET.fromstring(raw)

    def lname(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def val(el: ET.Element) -> str:
        # SEMO sometimes uses attributes like v="..."
        if "v" in el.attrib:
            return str(el.attrib["v"]).strip()
        return (el.text or "").strip()

    if debug:
        from collections import Counter
        tags = [lname(el.tag).lower() for el in root.iter()]
        print("[DEBUG] Top XML tags:", Counter(tags).most_common(30))

    # Find all TimeSeries blocks (case-insensitive)
    ts_blocks = [el for el in root.iter() if "timeseries" == lname(el.tag).lower()]
    if not ts_blocks:
        # fallback: sometimes "TimeSeries" is nested or slightly named
        ts_blocks = [el for el in root.iter() if "timeseries" in lname(el.tag).lower()]

    frames = []

    # Helper: parse one TimeSeries
    def parse_one_timeseries(ts_el: ET.Element) -> pd.DataFrame:
        bidding_area = None
        for el in ts_el.iter():
            if lname(el.tag).lower() == "biddingarea":
                bidding_area = val(el)
                break

        rows = []
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

            ts = pd.to_datetime(start_txt, utc=True, errors="coerce")
            try:
                pr = float(price_txt)
            except Exception:
                pr = None

            if ts is not pd.NaT and pr is not None:
                rows.append((ts, pr, bidding_area))

        if not rows:
            return pd.DataFrame(columns=["ts_utc", "dam_60min_hrp_eur_mwh", "bidding_area"])

        return pd.DataFrame(rows, columns=["ts_utc", "dam_60min_hrp_eur_mwh", "bidding_area"])

    if ts_blocks:
        for ts_el in ts_blocks:
            df = parse_one_timeseries(ts_el)
            if not df.empty:
                frames.append(df)
    else:
        # No TimeSeries blocks found -> parse Intervals globally (rare)
        rows = []
        for interval in root.iter():
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
            ts = pd.to_datetime(start_txt, utc=True, errors="coerce")
            try:
                pr = float(price_txt)
            except Exception:
                pr = None
            if ts is not pd.NaT and pr is not None:
                rows.append((ts, pr, None))
        if rows:
            frames.append(pd.DataFrame(rows, columns=["ts_utc", "dam_60min_hrp_eur_mwh", "bidding_area"]))

    if not frames:
        raise RuntimeError("Parsed XML but found no Interval(StartTime, MarketPriceRoundedAmount) rows.")

    out = pd.concat(frames, ignore_index=True)
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True, errors="coerce")
    out["dam_60min_hrp_eur_mwh"] = pd.to_numeric(out["dam_60min_hrp_eur_mwh"], errors="coerce")
    out = out.dropna(subset=["ts_utc", "dam_60min_hrp_eur_mwh"]).sort_values("ts_utc")

    # If there are duplicates (e.g., overlapping time series), keep last
    out = out.drop_duplicates(["ts_utc", "bidding_area"], keep="last").reset_index(drop=True)
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
    ap.add_argument("--debug", action="store_true", help="Print XML tag histogram when parsing")
    ap.add_argument("--bidding_area", default=None, help="Optional: filter to one bidding area code from the XML")
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
                df = _parse_hrp60_xml(raw, debug=args.debug)
            elif rn.endswith(".csv"):
                df = _parse_csv(raw)
            else:
                # try XML first, then CSV
                try:
                    df = _parse_hrp60_xml(raw, debug=args.debug)
                except Exception:
                    df = _parse_csv(raw)
        except Exception as e:
            print(f"[WARN] Failed parsing {item.resource_name} for {d_str}: {e}", file=sys.stderr)
            continue

        if args.bidding_area:
            df = df[df["bidding_area"].astype(str) == args.bidding_area]

        if args.debug and "bidding_area" in df.columns and not df.empty:
            print("[DEBUG] bidding_area values:", df["bidding_area"].dropna().unique().tolist())

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
