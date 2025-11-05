# src/data/semopx_api.py
from __future__ import annotations
import io, time, random
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

# From your DevTools/cURL (works as-is)
SEMO_DOC_ID_HRP = "69074b479620d95ac3217943"
SEMO_API_BASE   = "https://reports.sem-o.com"
SEMO_HEADERS = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "origin": "https://www.semopx.com",
    "referer": "https://www.semopx.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
}

def _sleep(attempt: int, base: float = 0.6, cap: float = 6.0):
    time.sleep(min(cap, base * (2 ** (attempt - 1)) + random.uniform(0, 0.3)))

def _list_hrp_files(doc_id: str) -> List[Dict]:
    url = f"{SEMO_API_BASE}/api/v1/documents/{doc_id}"
    last_err = None
    for a in range(1, 6):
        try:
            r = requests.get(url, params={"IST": "1"}, headers=SEMO_HEADERS, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "Documents" in data:
                return data["Documents"]
            if isinstance(data, list):
                return data
            raise RuntimeError(f"Unexpected SEMO JSON shape: {type(data)}")
        except Exception as e:
            last_err = e; _sleep(a)
    raise RuntimeError(f"Could not list SEMO HRP files: {last_err}")

def _best_match(items: List[Dict], d: date) -> Optional[Dict]:
    ymd = d.strftime("%Y-%m-%d")
    for it in items:
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
    # fallback to nearest
    dated = []
    for it in items:
        eff = it.get("EffectiveDate") or it.get("effectiveDate") or it.get("TradingDate")
        try: dated.append((abs((pd.Timestamp(eff).date() - d).days), it))
        except Exception: pass
    return sorted(dated, key=lambda x: x[0])[0][1] if dated else None

def _download(item: Dict) -> bytes:
    url = item.get("Url") or item.get("url")
    if url:
        if url.startswith("/"):
            url = f"{SEMO_API_BASE}{url}"
        last_err = None
        for a in range(1, 5):
            try:
                r = requests.get(url, headers=SEMO_HEADERS, timeout=60)
                r.raise_for_status()
                return r.content
            except Exception as e:
                last_err = e; _sleep(a)
        raise RuntimeError(f"Download failed (Url): {last_err}")

    doc_id = item.get("DocumentId") or item.get("documentId") or SEMO_DOC_ID_HRP
    fname  = item.get("FileName")  or item.get("fileName")
    if not fname:
        raise RuntimeError("Cannot assemble download URL: FileName missing.")
    url = f"{SEMO_API_BASE}/api/v1/documents/{doc_id}/download?filename={fname}"
    last_err = None
    for a in range(1, 5):
        try:
            r = requests.get(url, headers=SEMO_HEADERS, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e; _sleep(a)
    raise RuntimeError(f"Download failed (download endpoint): {last_err}")

def _parse_hrp_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    cols = {str(c).strip().lower(): c for c in df.columns}
    date_col  = cols.get("date") or cols.get("trading day") or list(df.columns)[0]
    time_col  = cols.get("time") or cols.get("hour") or list(df.columns)[1]
    price_col = (cols.get("eur/mwh") or cols.get("price (eur/mwh)") or
                 cols.get("dam price eur/mwh") or cols.get("price") or list(df.columns)[-1])

    ts_local = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    if ts_local.dt.tz is None:
        ts_local = ts_local.dt.tz_localize("Europe/Dublin", nonexistent="shift_forward", ambiguous="NaT")
    ts_utc = ts_local.dt.tz_convert("UTC")

    out = pd.DataFrame({
        "ts_utc": ts_utc,
        "dam_eur_mwh": pd.to_numeric(df[price_col], errors="coerce"),
    }).dropna(subset=["ts_utc"]).sort_values("ts_utc")
    out = out[~out["ts_utc"].duplicated(keep="last")]
    return out.reset_index(drop=True)

def fetch_dam_hrp_for_date(d: date, cache_dir: Path = Path("data/raw/semopx/hrp"), force: bool = False) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{d.strftime('%Y-%m-%d')}.csv"
    if fp.exists() and not force:
        return _parse_hrp_csv(fp.read_bytes())
    items = _list_hrp_files(SEMO_DOC_ID_HRP)
    item = _best_match(items, d)
    if not item:
        raise RuntimeError(f"No SEMO HRP file for {d}")
    raw = _download(item)
    fp.write_bytes(raw)
    return _parse_hrp_csv(raw)

def fetch_dam_hrp_recent(days: int = 21, cache_dir: Path = Path("data/raw/semopx/hrp"), force: bool = False) -> pd.DataFrame:
    from datetime import date as _date
    today = _date.today()
    dfs = []
    for k in range(days):
        d = today - timedelta(days=k)
        try:
            dfs.append(fetch_dam_hrp_for_date(d, cache_dir=cache_dir, force=force))
        except Exception:
            continue
    if not dfs:
        raise RuntimeError("No HRP files downloaded in the requested window.")
    out = pd.concat(dfs, ignore_index=True).sort_values("ts_utc")
    out = out[~out["ts_utc"].duplicated(keep="last")]
    return out.reset_index(drop=True)
