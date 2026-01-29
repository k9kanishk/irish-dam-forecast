# src/data/semopx_api_v2.py
"""
Improved SEMOpx API client for fetching Irish DAM prices.
Uses the official SEMOpx reports API to get the latest data.
"""
from __future__ import annotations
import io
import time
import random
import hashlib
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import requests

# SEMOpx API endpoints
SEMOPX_API_BASE = "https://reports.sem-o.com/api/v1"
SEMOPX_DOC_LIBRARY = "https://www.semopx.com/market-data/document-library"

# Document IDs for different report types (these may change - check SEMOpx website)
DOC_IDS = {
    "dam_results": "dam-market-results",  # Day-Ahead Market Results
    "ida1_results": "ida1-market-results",  # Intraday Auction 1
    "hrp": "harmonised-reference-price",  # Hourly Reference Prices
}

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "origin": "https://www.semopx.com",
    "referer": "https://www.semopx.com/",
}

CACHE_DIR = Path("data/raw/semopx_v2")


class SEMOpxClient:
    """Client for fetching data from SEMOpx API."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def _sleep_with_jitter(self, attempt: int, base: float = 0.5):
        """Exponential backoff with jitter."""
        delay = base * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(min(delay, 10))
    
    def _get_cache_path(self, report_type: str, date_str: str) -> Path:
        """Get cache file path for a specific report and date."""
        return self.cache_dir / f"{report_type}_{date_str}.parquet"
    
    def _fetch_report_list(self, report_type: str = "dam_results") -> list[dict]:
        """
        Fetch list of available reports from SEMOpx.
        Returns list of report metadata dicts.
        """
        # Try the documents endpoint
        url = f"{SEMOPX_API_BASE}/documents"
        params = {"type": report_type, "page": 1, "pageSize": 100}
        
        for attempt in range(5):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                if isinstance(data, dict):
                    return data.get("Documents", data.get("documents", []))
                return data
            except Exception as e:
                if attempt < 4:
                    self._sleep_with_jitter(attempt)
                else:
                    raise RuntimeError(f"Failed to fetch report list: {e}")
        
        return []
    
    def _download_report(self, doc_id: str, filename: str) -> bytes:
        """Download a specific report file."""
        url = f"{SEMOPX_API_BASE}/documents/{doc_id}/download"
        params = {"filename": filename}
        
        for attempt in range(5):
            try:
                resp = self.session.get(url, params=params, timeout=60)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                if attempt < 4:
                    self._sleep_with_jitter(attempt)
                else:
                    raise RuntimeError(f"Failed to download report: {e}")
        
        return b""
    
    def _parse_dam_csv(self, raw: bytes) -> pd.DataFrame:
        """Parse DAM results CSV into standardized DataFrame."""
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            # Try with different encodings
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        
        # Normalize column names
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        
        # Find timestamp and price columns
        ts_col = None
        price_col = None
        
        for col in df.columns:
            if any(x in col for x in ["timestamp", "datetime", "time", "delivery"]):
                ts_col = col
            if any(x in col for x in ["price", "eur", "€", "clearing"]):
                price_col = col
        
        if ts_col is None or price_col is None:
            raise ValueError(f"Could not identify columns. Found: {df.columns.tolist()}")
        
        # Parse timestamps (SEMOpx uses ISO format or local Dublin time)
        df["ts_utc"] = pd.to_datetime(df[ts_col], errors="coerce")
        
        # Convert to UTC if needed
        if df["ts_utc"].dt.tz is None:
            # Assume Dublin time if no timezone
            df["ts_utc"] = df["ts_utc"].dt.tz_localize(
                "Europe/Dublin", ambiguous="NaT", nonexistent="shift_forward"
            ).dt.tz_convert("UTC")
        elif str(df["ts_utc"].dt.tz) != "UTC":
            df["ts_utc"] = df["ts_utc"].dt.tz_convert("UTC")
        
        # Parse price
        df["dam_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
        
        # Clean up
        result = df[["ts_utc", "dam_eur_mwh"]].dropna()
        result = result.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
        
        return result.reset_index(drop=True)
    
    def fetch_dam_prices(
        self, 
        start_date: date, 
        end_date: date,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch DAM prices for a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            force_refresh: Force re-download even if cached
            
        Returns:
            DataFrame with columns: ts_utc, dam_eur_mwh
        """
        all_data = []
        
        # Iterate through each day
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            cache_path = self._get_cache_path("dam", date_str)
            
            if cache_path.exists() and not force_refresh:
                # Load from cache
                df = pd.read_parquet(cache_path)
                all_data.append(df)
            else:
                # Try to fetch from API
                try:
                    df = self._fetch_single_day(current)
                    if not df.empty:
                        df.to_parquet(cache_path)
                        all_data.append(df)
                except Exception as e:
                    print(f"Warning: Failed to fetch {date_str}: {e}")
            
            current += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
        return result.reset_index(drop=True)
    
    def _fetch_single_day(self, d: date) -> pd.DataFrame:
        """Fetch DAM prices for a single day."""
        # SEMOpx publishes results with specific naming patterns
        # Try different URL patterns
        
        patterns = [
            f"{SEMOPX_API_BASE}/files/dam-results/{d.strftime('%Y/%m')}/EA_HRP_{d.strftime('%Y%m%d')}.csv",
            f"{SEMOPX_API_BASE}/files/market-results/EA_DAMResults_{d.strftime('%Y%m%d')}.csv",
        ]
        
        for url in patterns:
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return self._parse_dam_csv(resp.content)
            except Exception:
                continue
        
        return pd.DataFrame()
    
    def fetch_recent(self, days: int = 30, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch last N days of DAM prices.
        
        Args:
            days: Number of days to fetch
            force_refresh: Force re-download
            
        Returns:
            DataFrame with columns: ts_utc, dam_eur_mwh
        """
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_dam_prices(start_date, end_date, force_refresh)


def fetch_dam_recent_v2(days: int = 30, force_refresh: bool = False) -> pd.DataFrame:
    """Convenience function to fetch recent DAM prices."""
    client = SEMOpxClient()
    return client.fetch_recent(days=days, force_refresh=force_refresh)


# Alternative: Direct market results page scraping
def fetch_from_market_results_page(days: int = 7) -> pd.DataFrame:
    """
    Fetch recent prices by scraping the SEMOpx market results page.
    This is a backup method if the API doesn't work.
    """
    url = "https://www.semopx.com/market-data/market-results"
    
    # Note: This would require Selenium or similar for JavaScript-rendered content
    # For now, return empty and recommend using the API or manual download
    
    print("Note: Market results page requires JavaScript. Use API or manual download.")
    return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


if __name__ == "__main__":
    # Test the client
    client = SEMOpxClient()
    
    print("Fetching last 7 days of DAM prices...")
    df = client.fetch_recent(days=7)
    
    if not df.empty:
        print(f"✅ Fetched {len(df)} price points")
        print(f"Date range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
        print(f"Price range: €{df['dam_eur_mwh'].min():.2f} - €{df['dam_eur_mwh'].max():.2f}/MWh")
    else:
        print("❌ No data fetched. Check if API endpoints have changed.")
        print("Recommendation: Download lookback files manually from SEMOpx")
