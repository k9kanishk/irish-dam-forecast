# src/data/entsoe_fixed.py
"""
Fixed ENTSO-E API client with better Ireland/SEM handling.

Key improvements:
1. Multiple area code fallbacks
2. Proper timezone handling for Ireland
3. Chunked fetching to avoid API limits
4. Better error messages
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
import pandas as pd
import pytz

from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Try to import entsoe-py
try:
    from entsoe import EntsoePandasClient
    from entsoe.exceptions import NoMatchingDataError
    ENTSOE_AVAILABLE = True
except ImportError:
    ENTSOE_AVAILABLE = False
    print("Warning: entsoe-py not installed. Run: pip install entsoe-py")

# Ireland/SEM area codes to try (in order of preference)
IRELAND_AREA_CODES = [
    "IE",                    # Standard Ireland code
    "IE_SEM",               # Ireland in SEM
    "10Y1001A1001A59C",     # SEM bidding zone EIC
    "10YIE-1001A00010",     # Ireland EIC
]

# Timezone
DUBLIN_TZ = ZoneInfo("Europe/Dublin")
BRUSSELS_TZ = ZoneInfo("Europe/Brussels")  # ENTSO-E uses Brussels
UTC = ZoneInfo("UTC")

CACHE_DIR = Path("data/raw/entsoe")


class EntsoeClientFixed:
    """
    Fixed ENTSO-E client with better Ireland support.
    """
    
    def __init__(self, token: Optional[str] = None):
        if not ENTSOE_AVAILABLE:
            raise RuntimeError("entsoe-py not installed")
        
        self.token = token or os.getenv("ENTSOE_TOKEN")
        if not self.token:
            raise RuntimeError(
                "ENTSOE_TOKEN not found. Set it in .env or pass to constructor.\n"
                "Get a token at: https://transparency.entsoe.eu/ (register → email transparency@entsoe.eu)"
            )
        
        self.client = EntsoePandasClient(api_key=self.token)
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _to_brussels(self, dt) -> pd.Timestamp:
        """Convert datetime to Brussels timezone (ENTSO-E standard)."""
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize(BRUSSELS_TZ)
        else:
            ts = ts.tz_convert(BRUSSELS_TZ)
        return ts
    
    def _query_with_fallback(self, query_func, start, end) -> Optional[pd.Series]:
        """
        Try query with multiple area codes until one works.
        """
        start_bz = self._to_brussels(start)
        end_bz = self._to_brussels(end)
        
        errors = []
        
        for area in IRELAND_AREA_CODES:
            try:
                result = query_func(area, start=start_bz, end=end_bz)
                
                if result is not None and len(result) > 0:
                    # Convert to UTC
                    if hasattr(result, 'tz_convert'):
                        result = result.tz_convert("UTC")
                    elif hasattr(result, 'index') and hasattr(result.index, 'tz_convert'):
                        result.index = result.index.tz_convert("UTC")
                    
                    return result
                    
            except NoMatchingDataError:
                errors.append(f"{area}: No matching data")
            except Exception as e:
                errors.append(f"{area}: {type(e).__name__}: {str(e)[:50]}")
        
        # Log all errors for debugging
        print(f"All area codes failed for {start} to {end}:")
        for err in errors:
            print(f"  - {err}")
        
        return None
    
    def fetch_day_ahead_prices(
        self,
        start_date: str | datetime | pd.Timestamp,
        end_date: str | datetime | pd.Timestamp,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch day-ahead prices for Ireland/SEM.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            cache: Whether to use/save cache
            
        Returns:
            DataFrame with columns: dam_eur_mwh, indexed by UTC datetime
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Cache key
        cache_key = f"dam_ie_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        
        # Try cache first
        if cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            if not df.empty:
                return df
        
        # Fetch from API
        def query_da(area, start, end):
            return self.client.query_day_ahead_prices(area, start=start, end=end)
        
        # ENTSO-E end is exclusive, add 1 day
        result = self._query_with_fallback(query_da, start, end + pd.Timedelta(days=1))
        
        if result is None or len(result) == 0:
            return pd.DataFrame(columns=["dam_eur_mwh"])
        
        # Convert to DataFrame
        df = result.to_frame("dam_eur_mwh")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        
        # Ensure UTC index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        
        # Save to cache
        if cache:
            df.to_parquet(cache_path)
        
        return df
    
    def fetch_recent_chunked(
        self,
        days: int = 30,
        chunk_days: int = 7,
        delay_days: int = 2
    ) -> pd.DataFrame:
        """
        Fetch recent prices in chunks to avoid API limits.
        
        Args:
            days: Total days to fetch
            chunk_days: Days per chunk
            delay_days: Skip most recent N days (may not be available yet)
            
        Returns:
            DataFrame with DAM prices
        """
        # Calculate date range
        end = datetime.now(DUBLIN_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=delay_days)
        start = end - timedelta(days=days)
        
        # Build chunks
        chunks = []
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            chunks.append((current, chunk_end))
            current = chunk_end
        
        # Fetch each chunk
        results = []
        for chunk_start, chunk_end in chunks:
            try:
                df = self.fetch_day_ahead_prices(
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d")
                )
                if not df.empty:
                    results.append(df)
            except Exception as e:
                print(f"Warning: Failed to fetch chunk {chunk_start} to {chunk_end}: {e}")
        
        if not results:
            return pd.DataFrame(columns=["dam_eur_mwh"])
        
        # Combine chunks
        df = pd.concat(results).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        
        return df
    
    def fetch_load_forecast(
        self,
        start_date: str | datetime,
        end_date: str | datetime
    ) -> pd.Series:
        """Fetch load forecast for Ireland."""
        def query_load(area, start, end):
            df = self.client.query_load_and_forecast(area, start=start, end=end)
            # Find forecast column
            for col in df.columns:
                col_str = str(col).lower()
                if "forecast" in col_str:
                    return df[col]
            return df.iloc[:, -1]  # Last column as fallback
        
        result = self._query_with_fallback(
            query_load,
            start_date,
            pd.Timestamp(end_date) + pd.Timedelta(days=1)
        )
        
        if result is None:
            return pd.Series(dtype=float, name="load_forecast_mw")
        
        result.name = "load_forecast_mw"
        return result
    
    def fetch_wind_solar_forecast(
        self,
        start_date: str | datetime,
        end_date: str | datetime
    ) -> pd.DataFrame:
        """Fetch wind and solar forecasts for Ireland."""
        psr_types = {
            "B16": "solar_mw",
            "B18": "wind_offshore_mw", 
            "B19": "wind_onshore_mw",
        }
        
        results = {}
        for psr, name in psr_types.items():
            def query_ws(area, start, end, psr_type=psr):
                return self.client.query_wind_and_solar_forecast(
                    area, start=start, end=end, psr_type=psr_type
                )
            
            result = self._query_with_fallback(
                query_ws,
                start_date,
                pd.Timestamp(end_date) + pd.Timedelta(days=1)
            )
            
            if result is not None and len(result) > 0:
                results[name] = result
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Add total wind
        wind_cols = [c for c in df.columns if "wind" in c]
        if wind_cols:
            df["wind_total_mw"] = df[wind_cols].sum(axis=1, min_count=1)
        
        return df


def fetch_ie_dam_recent(days: int = 30) -> pd.DataFrame:
    """Convenience function to fetch recent Irish DAM prices from ENTSO-E."""
    try:
        client = EntsoeClientFixed()
        return client.fetch_recent_chunked(days=days)
    except Exception as e:
        print(f"ENTSO-E fetch failed: {e}")
        return pd.DataFrame(columns=["dam_eur_mwh"])


def check_entsoe_connection() -> dict:
    """
    Test ENTSO-E connection and return status.
    """
    result = {
        "token_set": bool(os.getenv("ENTSOE_TOKEN")),
        "library_available": ENTSOE_AVAILABLE,
        "connection_ok": False,
        "data_available": False,
        "error": None,
    }
    
    if not result["token_set"]:
        result["error"] = "ENTSOE_TOKEN not set in environment"
        return result
    
    if not result["library_available"]:
        result["error"] = "entsoe-py library not installed"
        return result
    
    try:
        client = EntsoeClientFixed()
        
        # Try to fetch yesterday's data
        yesterday = datetime.now() - timedelta(days=2)
        df = client.fetch_day_ahead_prices(
            yesterday.strftime("%Y-%m-%d"),
            yesterday.strftime("%Y-%m-%d"),
            cache=False
        )
        
        result["connection_ok"] = True
        result["data_available"] = not df.empty
        
        if df.empty:
            result["error"] = "API responded but no data for Ireland/SEM"
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


if __name__ == "__main__":
    print("Checking ENTSO-E connection...")
    status = check_entsoe_connection()
    
    print(f"\nStatus:")
    print(f"  Token set: {status['token_set']}")
    print(f"  Library available: {status['library_available']}")
    print(f"  Connection OK: {status['connection_ok']}")
    print(f"  Data available: {status['data_available']}")
    
    if status["error"]:
        print(f"  Error: {status['error']}")
    
    if status["data_available"]:
        print("\n✅ ENTSO-E is working! Fetching last 7 days...")
        client = EntsoeClientFixed()
        df = client.fetch_recent_chunked(days=7)
        print(f"  Got {len(df)} price points")
        print(f"  Range: {df.index.min()} to {df.index.max()}")
