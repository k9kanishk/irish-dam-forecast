# src/data/entsoe_api.py - FIXED VERSION
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import pytz
from zoneinfo import ZoneInfo
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from datetime import timedelta

TZ_QUERY = ZoneInfo("Europe/Brussels")

def fetch_ie_dam_chunked(days: int = 21, chunk_days: int = 7, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch last `days` in chunks (7d by default). Faster + fewer timeouts.
    Automatically handles ENTSO-E delays by stepping back if recent data unavailable.
    """
    tz = pytz.timezone("Europe/Dublin")
    
    # CRITICAL FIX: Start with dates that are more likely to have data
    # ENTSO-E typically has 2-5 day delay
    end_local = pd.Timestamp.now(tz).normalize() - pd.Timedelta(days=3)  # Go back 3 days from today
    start_local = end_local - pd.Timedelta(days=days)

    # Build chunk windows
    windows = []
    cur = start_local
    while cur < end_local:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_local)
        windows.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt

    parts = []
    failed_chunks = []
    
    # Sequential fetching with error handling per chunk
    for s, e in windows:
        try:
            chunk_df = fetch_ie_dam_prices_entsoe(s, e, force_refresh=force_refresh)
            if not chunk_df.empty:
                parts.append(chunk_df)
            else:
                failed_chunks.append((s, e))
        except Exception as ex:
            failed_chunks.append((s, e))
            print(f"Warning: Failed to fetch chunk {s} to {e}: {ex}")

    if not parts:
        # If all chunks failed, try an even older date range
        print(f"All chunks failed. Trying older date range...")
        end_local = pd.Timestamp.now(tz).normalize() - pd.Timedelta(days=7)
        start_local = end_local - pd.Timedelta(days=min(days, 30))
        
        try:
            fallback = fetch_ie_dam_prices_entsoe(
                start_local.strftime("%Y-%m-%d"),
                end_local.strftime("%Y-%m-%d"),
                force_refresh=True
            )
            if not fallback.empty:
                return fallback
        except:
            pass
        
        raise RuntimeError(
            f"Could not fetch any data from ENTSO-E. "
            f"Tried dates from {start_local.date()} to {end_local.date()}. "
            f"ENTSO-E typically has 2-5 day delay. Please try with older dates."
        )

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_ie_dam_prices_entsoe(
    start_date: str,
    end_date: str,
    cache_dir: Path = Path("data/processed"),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch IE day-ahead prices from ENTSO-E and return a UTC-indexed df with 'dam_eur_mwh'.
    Handles NoMatchingDataError by trying progressively older dates.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"dam_ie_{start_date}_{end_date}.parquet"

    if fp.exists() and not force_refresh:
        try:
            df = pd.read_parquet(fp)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df["dam_eur_mwh"] = pd.to_numeric(df["dam_eur_mwh"], errors="coerce")
            return df
        except Exception:
            pass  # Cache read failed, fetch fresh

    token = os.getenv("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_TOKEN not set.")

    client = EntsoePandasClient(api_key=token, timeout=30)
    tz = pytz.timezone("Europe/Dublin")
    
    # Try the requested dates first
    start = tz.localize(pd.Timestamp(start_date))
    end = tz.localize(pd.Timestamp(end_date)) + pd.Timedelta(days=1)

    # Try with progressively older dates if NoMatchingDataError
    for days_back in range(0, 8):  # Try up to 7 days earlier
        try:
            adjusted_start = start - pd.Timedelta(days=days_back)
            adjusted_end = end - pd.Timedelta(days=days_back)
            
            ser = client.query_day_ahead_prices("IE", start=adjusted_start, end=adjusted_end)
            
            if ser is not None and len(ser) > 0:
                if days_back > 0:
                    print(f"Note: Got data for adjusted dates ({days_back} days earlier)")
                
                ser = ser.tz_convert("UTC")
                df = ser.to_frame("dam_eur_mwh")
                df = df[~df.index.duplicated(keep="last")].sort_index()
                
                # Cache the result
                df.to_parquet(fp)
                return df
                
        except NoMatchingDataError:
            if days_back == 7:  # Last attempt
                # Return empty DataFrame rather than raising
                print(f"Warning: No ENTSO-E data available for {start_date} to {end_date} (tried up to 7 days back)")
                return pd.DataFrame(columns=["dam_eur_mwh"])
            continue  # Try next day back
        except Exception as e:
            if days_back == 7:
                print(f"Warning: ENTSO-E fetch failed: {e}")
                return pd.DataFrame(columns=["dam_eur_mwh"])
            continue

    # If we get here, return empty
    return pd.DataFrame(columns=["dam_eur_mwh"])


def fetch_ie_dam_recent(days: int = 21, force_refresh: bool = True) -> pd.DataFrame:
    """
    Return last `days` of IE prices (UTC-indexed df).
    Uses safe date range accounting for ENTSO-E delay.
    """
    tz = pytz.timezone("Europe/Dublin")
    today = pd.Timestamp.now(tz).normalize()
    
    # CRITICAL: Go back 3 days from today to avoid NoMatchingDataError
    end = today - pd.Timedelta(days=3)
    start = end - pd.Timedelta(days=days)
    
    return fetch_ie_dam_prices_entsoe(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        force_refresh=force_refresh
    )


class Entsoe:
    """Small wrapper to fetch other fundamentals consistently."""
    def __init__(self, token: str | None = None, area: str = "IE"):
        token = token or os.getenv("ENTSOE_TOKEN")
        if not token:
            raise RuntimeError("ENTSOE_TOKEN not set")
        self.client = EntsoePandasClient(api_key=token, timeout=30)
        self.area = area

    def _brussels(self, dt_like) -> pd.Timestamp:
        ts = pd.Timestamp(dt_like)
        if ts.tzinfo is None:
            return ts.tz_localize(TZ_QUERY)
        return ts.tz_convert(TZ_QUERY)

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        """Fetch DAM prices with automatic fallback to older dates"""
        try:
            # Try original dates
            s = self.client.query_day_ahead_prices(
                self.area,
                start=self._brussels(start),
                end=self._brussels(end)
            )
            if s is not None and len(s) > 0:
                s.name = "dam_eur_mwh"
                return s
        except NoMatchingDataError:
            pass  # Try older dates
        except Exception as e:
            print(f"Warning: DAM fetch failed: {e}")
            
        # Try stepping back up to 7 days
        for days_back in range(1, 8):
            try:
                adjusted_end = pd.Timestamp(end) - timedelta(days=days_back)
                s = self.client.query_day_ahead_prices(
                    self.area,
                    start=self._brussels(start),
                    end=self._brussels(adjusted_end)
                )
                if s is not None and len(s) > 0:
                    print(f"Got data up to {adjusted_end.date()} ({days_back} days back)")
                    s.name = "dam_eur_mwh"
                    return s
            except:
                continue
        
        # Return empty if all fails
        return pd.Series(dtype=float, name="dam_eur_mwh")

    def load_forecast(self, start: str, end: str) -> pd.Series:
        """Fetch load forecast with error handling"""
        try:
            df = self.client.query_load_and_forecast(
                self.area,
                start=self._brussels(start),
                end=self._brussels(end)
            )
            
            def _norm(c):
                try:
                    return " ".join(map(str, c)).lower()
                except TypeError:
                    return str(c).lower()
            
            cand = [c for c in df.columns if "forecast" in _norm(c)]
            col = cand[0] if cand else df.columns[-1]
            s = pd.to_numeric(df[col], errors="coerce")
            s.name = "load_forecast_mw"
            return s
        except Exception as e:
            print(f"Warning: Load forecast fetch failed: {e}")
            return pd.Series(dtype=float, name="load_forecast_mw")

    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        """Fetch wind/solar forecast with error handling"""
        def _try(psr: str) -> pd.Series:
            try:
                s = self.client.query_wind_and_solar_forecast(
                    self.area,
                    start=self._brussels(start),
                    end=self._brussels(end),
                    psr_type=psr
                )
                return s if s is not None else pd.Series(dtype=float)
            except (NoMatchingDataError, Exception):
                return pd.Series(dtype=float)

        solar = _try("B16")
        wind_on = _try("B19")
        wind_off = _try("B18")

        df = pd.concat(
            {"solar_mw": solar, "wind_onshore_mw": wind_on, "wind_offshore_mw": wind_off},
            axis=1
        ).sort_index()

        cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df]
        if cols:
            df["wind_total_mw"] = df[cols].sum(axis=1, min_count=1)
        return df
