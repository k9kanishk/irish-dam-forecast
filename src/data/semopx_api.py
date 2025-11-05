# src/data/semopx_api.py
from __future__ import annotations
import io, os, time, random
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

# === Config (from your cURL) ==================================================
SEMO_DOC_ID_HRP = os.getenv("SEMO_DOC_ID_HRP", "69074b479620d95ac3217943")  # DAM 60min HRP
SEMO_API_BASE   = "https://reports.sem-o.com"
# Minimal headers (exactly from your cURL; keep user-agent at least)
SEMO_HEADERS = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "origin": "https://www.semopx.com",
    "referer": "https://www.semopx.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
}

# === Internal helpers =========================================================
def _sleep_backoff(attempt: int, base: float = 0.6, cap: float = 6.0):
    time.sleep(min(cap, base * (2 ** (attempt - 1)) + random.uniform(0, 0.3)))

def _list_hrp_files(doc_id: str) -> List[Dict]:
    """List files for the HRP document. Returns items from the JSON array."""
    url = f"{SEMO_API_BASE}/api/v1/documents/{doc_id}"
    last_err = None
    for attempt in range(1, 6):
        try:
            r = requests.get(url, params={"IST": "1"}, headers=SEMO_HEADERS, timeout=60)
            r.raise_for_status()
            data = r.json()
            # API returns a list of files or an object with a 'Documents' field (varies over time).
            if isinstance(data, dict) and "Documents" in data:
                return data["Documents"]
            if isinstance(data, list):
                return data
            raise RuntimeError(f"Unexpected SEMO JSON shape: {type(data)}")
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt)
    raise RuntimeError(f"Could not list SEMO HRP files: {last_err}")

def _best_match_for_date(items: List[Dict], d: date) -> Optional[Dict]:
    """
    Try to find the file entry for the given civil date.
    We match on any of: 'EffectiveDate', 'TradingDate', filename or URL containing YYYY-MM-DD.
    """
    ymd = d.strftime("%Y-%m-%d")
    for it in items:
        # Common fields seen historically:
        eff = it.get("EffectiveDate") or it.get("effectiveDate") or it.get("TradingDate")
        if eff:
            try:
                if pd.Timestamp(eff).date() == d:
                    return it
            except Exception:
                pass
        fname = (it.get("FileName") or it.get("fileName") or "")
        url   = (it.get("Url") or it.get("url") or "")
        if ymd in fname or ymd in url:
            return it
    # If not found, try nearest previous file (common if today's isn’t out yet)
    dated_items = []
    for it in items:
        eff = it.get("EffectiveDate") or it.get("effectiveDate") or it.get("TradingDate")
        try:
            dated_items.append((abs((pd.Timestamp(eff).date() - d).days), it))
        except Exception:
            continue
    if dated_items:
        return sorted(dated_items, key=lambda x: x[0])[0][1]
    return None

def _download_file(item: Dict) -> bytes:
    """
    Download the actual CSV/ZIP using the item's URL.
    The JSON usually exposes either:
      - absolute '/documents/...' url, or
      - requires '/api/v1/documents/{id}/download?filename=...'
    We handle both.
    """
    # Prefer direct Url if present
    url = item.get("Url") or item.get("url")
    if url:
        if url.startswith("/"):
            url = f"{SEMO_API_BASE}{url}"
        last_err = None
        for attempt in range(1, 5):
            try:
                r = requests.get(url, headers=SEMO_HEADERS, timeout=60)
                r.raise_for_status()
                return r.content
            except Exception as e:
                last_err = e
                _sleep_backoff(attempt)
        raise RuntimeError(f"Download failed (Url): {last_err}")

    # Fallback: build the 'download' endpoint using document id + filename
    doc_id = item.get("DocumentId") or item.get("documentId") or SEMO_DOC_ID_HRP
    fname  = item.get("FileName")  or item.get("fileName")
    if not fname:
        raise RuntimeError("Cannot assemble download URL: FileName missing.")
    url = f"{SEMO_API_BASE}/api/v1/documents/{doc_id}/download?filename={fname}"
    last_err = None
    for attempt in range(1, 5):
        try:
            r = requests.get(url, headers=SEMO_HEADERS, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt)
    raise RuntimeError(f"Download failed (download endpoint): {last_err}")

def _parse_hrp_csv(raw: bytes) -> pd.DataFrame:
    """
    Parse the HRP CSV into a tidy frame with:
      ts_utc (UTC hourly), dam_eur_mwh (float)
    Adjust the column names if SEMO changes headers — we try to auto-detect.
    """
    df = pd.read_csv(io.BytesIO(raw))
    cols = {str(c).strip().lower(): c for c in df.columns}
    # Common guesses
    date_col = cols.get("date") or cols.get("trading day") or list(df.columns)[0]
    time_col = cols.get("time") or cols.get("hour") or list(df.columns)[1]
    price_col = (cols.get("eur/mwh") or cols.get("price (eur/mwh)") or
                 cols.get("dam price eur/mwh") or cols.get("price") or list(df.columns)[-1])

    # Build timestamp in Europe/Dublin, then convert to UTC
    ts_local = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str),
                              errors="coerce")
    # Some files include timezone info already; if naive, localize to Europe/Dublin
    if ts_local.dt.tz is None:
        ts_local = ts_local.dt.tz_localize("Europe/Dublin", nonexistent="shift_forward", ambiguous="NaT")
    ts_utc = ts_local.dt.tz_convert("UTC")

    out = pd.DataFrame({
        "ts_utc": ts_utc,
        "dam_eur_mwh": pd.to_numeric(df[price_col], errors="coerce"),
    }).dropna(subset=["ts_utc"]).sort_values("ts_utc")
    # Drop duplicates and enforce hourly
    out = out[~out["ts_utc"].duplicated(keep="last")]
    return out.reset_index(drop=True)

# === Public functions =========================================================
def fetch_dam_hrp_for_date(d: date,
                           cache_dir: Path = Path("data/raw/semopx/hrp"),
                           force: bool = False) -> pd.DataFrame:
    """
    Download + cache one trading day's HRP (DAM 60m) as tidy dataframe.
    Cache filename: <YYYY-MM-DD>.csv
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{d.strftime('%Y-%m-%d')}.csv"
    if cache_path.exists() and not force:
        raw = cache_path.read_bytes()
        return _parse_hrp_csv(raw)

    items = _list_hrp_files(SEMO_DOC_ID_HRP)
    item = _best_match_for_date(items, d)
    if not item:
        raise RuntimeError(f"No SEMO HRP file found for {d}")

    raw = _download_file(item)
    cache_path.write_bytes(raw)
    return _parse_hrp_csv(raw)

def fetch_dam_hrp_recent(days: int = 21,
                         cache_dir: Path = Path("data/raw/semopx/hrp"),
                         force: bool = False) -> pd.DataFrame:
    """Concatenate the last N civil days (today inclusive) of HRP."""
    today = date.today()
    dfs = []
    for k in range(days):
        d = today - timedelta(days=k)
        try:
            dfs.append(fetch_dam_hrp_for_date(d, cache_dir=cache_dir, force=force))
        except Exception:
            # skip missing days (e.g., if report not yet published)
            continue
    if not dfs:
        raise RuntimeError("No HRP files downloaded in the requested window.")
    out = pd.concat(dfs, ignore_index=True).sort_values("ts_utc")
    out = out[~out["ts_utc"].duplicated(keep="last")]
    return out.reset_index(drop=True)
