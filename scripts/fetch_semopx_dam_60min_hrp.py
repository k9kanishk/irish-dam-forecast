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
    Robust parser for SEMOpx DAM 60Min Harmonised Reference Price XML.

    Handles:
    - No <TimeSeries> present
    - <Period> blocks anywhere in the document
    - Points with either:
        a) explicit datetime per point, OR
        b) position + period start + resolution

    Returns:
      ts_utc (UTC), dam_60min_hrp_eur_mwh (float)
    """
    root = ET.fromstring(raw)

    def lname(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    # Collect all tags if debug
    if debug:
        tags = [lname(el.tag).lower() for el in root.iter()]
        from collections import Counter
        top = Counter(tags).most_common(30)
        print("[DEBUG] Top XML tags:", top)

    # Find ALL Period elements anywhere (not just inside TimeSeries)
    periods = [el for el in root.iter() if lname(el.tag).lower() == "period"]
    if not periods:
        # Some SEMO files use "PeriodTimeInterval" or similar naming
        periods = [el for el in root.iter() if "period" in lname(el.tag).lower()]

    def parse_resolution(res_txt: str | None) -> pd.Timedelta:
        if not res_txt:
            return pd.Timedelta(hours=1)
        r = res_txt.strip().upper()
        if r in ("PT60M", "PT1H"):
            return pd.Timedelta(hours=1)
        if r == "PT30M":
            return pd.Timedelta(minutes=30)
        # fallback
        return pd.Timedelta(hours=1)

    def extract_text(el: ET.Element) -> str:
        return (el.text or "").strip()

    def find_first_datetime(node: ET.Element) -> Optional[pd.Timestamp]:
        """
        Search descendants for something that looks like a timestamp.
        """
        for el in node.iter():
            n = lname(el.tag).lower()
            if any(k in n for k in ["datetime", "timestamp", "time", "start"]):
                txt = extract_text(el)
                # accept ISO strings like 2026-01-13T23:00:00Z
                ts = pd.to_datetime(txt, utc=True, errors="coerce")
                if not pd.isna(ts):
                    return ts
        return None

    def find_first_float(node: ET.Element) -> Optional[float]:
        """
        Search descendants for a numeric value in tags that look like price fields.
        """
        for el in node.iter():
            n = lname(el.tag).lower()
            if any(k in n for k in ["price", "amount", "eur", "mwh", "reference"]):
                txt = extract_text(el)
                try:
                    return float(txt)
                except Exception:
                    continue
        return None

    frames = []

    for period in periods:
        # find period start + resolution (where possible)
        start_ts = None
        step = pd.Timedelta(hours=1)

        # Many SEMO docs store values in attributes v="..."
        def get_v_or_text(el: ET.Element) -> str:
            if "v" in el.attrib:
                return str(el.attrib["v"]).strip()
            return extract_text(el)

        for el in period.iter():
            n = lname(el.tag).lower()
            if n == "start":
                start_ts = pd.to_datetime(get_v_or_text(el), utc=True, errors="coerce")
            elif "resolution" in n:
                step = parse_resolution(get_v_or_text(el))

        # Identify "point-like" nodes
        point_nodes = []
        for el in period:
            n = lname(el.tag).lower()
            if n in ("point", "row", "entry", "interval"):
                point_nodes.append(el)

        # If points are nested deeper
        if not point_nodes:
            point_nodes = [
                el for el in period.iter()
                if lname(el.tag).lower() in ("point", "row", "entry", "interval")
            ]

        rows = []

        for pt in point_nodes:
            # Try explicit timestamp in point first
            ts = find_first_datetime(pt)

            # Try position-based time if no explicit timestamp
            pos = None
            if ts is None and start_ts is not None:
                for el in pt.iter():
                    n = lname(el.tag).lower()
                    if n in ("position", "seq", "sequence", "index"):
                        txt = get_v_or_text(el)
                        try:
                            pos = int(txt)
                        except Exception:
                            pos = None
                        break
                if pos is not None:
                    ts = start_ts + (pos - 1) * step

            price = find_first_float(pt)

            if ts is not None and price is not None:
                rows.append((ts, price))

        if rows:
            tmp = pd.DataFrame(rows, columns=["ts_utc", "dam_60min_hrp_eur_mwh"])
            frames.append(tmp)

    if not frames:
        # Last-resort: scan ALL nodes for (datetime, price) pairs by proximity
        # (Better than returning empty)
        all_nodes = list(root.iter())
        pairs = []
        for node in all_nodes:
            ts = find_first_datetime(node)
            pr = find_first_float(node)
            if ts is not None and pr is not None:
                pairs.append((ts, pr))
        if pairs:
            out = pd.DataFrame(pairs, columns=["ts_utc", "dam_60min_hrp_eur_mwh"])
            out = out.dropna().sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
            out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
            out["dam_60min_hrp_eur_mwh"] = pd.to_numeric(out["dam_60min_hrp_eur_mwh"], errors="coerce")
            return out.reset_index(drop=True)

        raise RuntimeError(
            "Parsed XML but could not find any timestamp/price pairs. "
            "Run with --debug to inspect tags."
        )

    out = pd.concat(frames, ignore_index=True)
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True, errors="coerce")
    out["dam_60min_hrp_eur_mwh"] = pd.to_numeric(out["dam_60min_hrp_eur_mwh"], errors="coerce")
    out = out.dropna().sort_values("ts_utc")
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
    ap.add_argument("--debug", action="store_true", help="Print XML tag histogram when parsing")
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
