# src/data/semopx_scraper.py
"""
Automated SEMOpx Day-Ahead Market price fetcher.
Scrapes prices from the Market Results page or underlying API.
"""
from __future__ import annotations
import json
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import requests

# === Configuration ===
CACHE_DIR = Path("data/raw/semopx_daily")
COMBINED_FILE = Path("data/raw/dam_prices_combined.parquet")

# SEMOpx API endpoints (discovered from network inspection)
SEMOPX_API_BASE = "https://www.semopx.com/api"
MARKET_RESULTS_API = "https://www.semopx.com/api/market-data/market-results"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.semopx.com",
    "Referer": "https://www.semopx.com/market-data/market-results",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def _retry_request(url: str, params: dict = None, max_retries: int = 3) -> requests.Response:
    """Make request with exponential backoff retry."""
    last_error = None
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.5, 1.5) * (attempt + 1))
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1} failed: {e}")
    raise RuntimeError(f"All {max_retries} attempts failed: {last_error}")


def fetch_dam_for_date(target_date: datetime.date, currency: str = "EUR") -> pd.DataFrame:
    """
    Fetch Day-Ahead prices for a specific date from SEMOpx.
    
    Returns DataFrame with columns: ts_utc, dam_eur_mwh
    """
    date_str = target_date.strftime("%Y-%m-%d")
    
    # Try the Market Results API endpoint
    # The website makes requests like: /api/market-data/market-results?date=2025-12-31&report=day-ahead&currency=EUR
    params = {
        "date": date_str,
        "report": "day-ahead",
        "currency": currency,
    }
    
    try:
        resp = _retry_request(MARKET_RESULTS_API, params=params)
        data = resp.json()
        
        # Parse the response - structure may vary
        records = []
        
        # Handle different possible response structures
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("data") or data.get("rows") or data.get("results") or []
        else:
            rows = []
        
        for row in rows:
            # Extract timestamp and price
            # Common field names from SEMOpx
            ts_raw = (
                row.get("deliveryStart") or 
                row.get("timestamp") or 
                row.get("dateTime") or
                row.get("time") or
                f"{row.get('date', date_str)} {row.get('time', '00:00')}"
            )
            
            price = (
                row.get("price") or 
                row.get("eurMwh") or 
                row.get("EUR/MWh") or
                row.get("dam_price") or
                row.get("value")
            )
            
            if ts_raw and price is not None:
                try:
                    # Parse timestamp
                    if isinstance(ts_raw, str):
                        ts = pd.to_datetime(ts_raw)
                    else:
                        ts = pd.Timestamp(ts_raw)
                    
                    # Localize to Dublin then convert to UTC
                    if ts.tz is None:
                        ts = ts.tz_localize("Europe/Dublin", ambiguous="NaT", nonexistent="shift_forward")
                    ts_utc = ts.tz_convert("UTC")
                    
                    records.append({
                        "ts_utc": ts_utc,
                        "dam_eur_mwh": float(price)
                    })
                except Exception as e:
                    print(f"  Skipping row: {e}")
                    continue
        
        if not records:
            print(f"  No records parsed for {date_str}")
            return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
        
        df = pd.DataFrame(records)
        df = df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
        return df.reset_index(drop=True)
        
    except Exception as e:
        print(f"  API fetch failed for {date_str}: {e}")
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


def fetch_dam_alternative(target_date: datetime.date) -> pd.DataFrame:
    """
    Alternative method: scrape from the rendered page using requests-html or selenium.
    This is a fallback if the API method doesn't work.
    """
    # Try fetching the page and parsing any embedded JSON data
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"https://www.semopx.com/market-data/market-results?date={date_str}"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        
        # Look for embedded JSON in the page
        import re
        
        # Common patterns for embedded data
        patterns = [
            r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
            r'window\.__DATA__\s*=\s*({.*?});',
            r'"marketData"\s*:\s*(\[.*?\])',
            r'"prices"\s*:\s*(\[.*?\])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, resp.text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    # Process embedded data...
                    print(f"  Found embedded data with pattern")
                    break
                except json.JSONDecodeError:
                    continue
        
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
        
    except Exception as e:
        print(f"  Alternative fetch failed: {e}")
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


def fetch_dam_range(
    start_date: datetime.date,
    end_date: datetime.date,
    cache: bool = True
) -> pd.DataFrame:
    """
    Fetch DAM prices for a date range, with optional caching.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    all_frames = []
    current = start_date
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        cache_file = CACHE_DIR / f"{date_str}.parquet"
        
        # Check cache first
        if cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    print(f"  {date_str}: loaded from cache ({len(df)} rows)")
                    all_frames.append(df)
                    current += timedelta(days=1)
                    continue
            except Exception:
                pass
        
        # Fetch from API
        print(f"  {date_str}: fetching from SEMOpx...")
        df = fetch_dam_for_date(current)
        
        # Try alternative if primary failed
        if df.empty:
            df = fetch_dam_alternative(current)
        
        if not df.empty:
            print(f"  {date_str}: got {len(df)} rows")
            # Save to cache
            if cache:
                df.to_parquet(cache_file, index=False)
            all_frames.append(df)
        else:
            print(f"  {date_str}: no data available")
        
        current += timedelta(days=1)
        time.sleep(random.uniform(1, 2))  # Be nice to the server
    
    if not all_frames:
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
    
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
    return combined.reset_index(drop=True)


def fetch_recent_dam(days: int = 30, force_refresh: bool = False) -> pd.DataFrame:
    """
    Convenience function: fetch last N days of DAM prices.
    """
    end_date = datetime.now().date() - timedelta(days=1)  # Yesterday (today's prices may not be final)
    start_date = end_date - timedelta(days=days)
    
    print(f"Fetching DAM prices from {start_date} to {end_date}...")
    
    df = fetch_dam_range(start_date, end_date, cache=not force_refresh)
    
    # Save combined file
    if not df.empty:
        COMBINED_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(COMBINED_FILE, index=False)
        print(f"Saved {len(df)} rows to {COMBINED_FILE}")
    
    return df


def load_combined_dam() -> pd.DataFrame:
    """Load the combined DAM prices file."""
    if COMBINED_FILE.exists():
        df = pd.read_parquet(COMBINED_FILE)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        return df
    return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


# === Selenium-based scraper (more reliable but requires browser) ===

def fetch_dam_selenium(target_date: datetime.date) -> pd.DataFrame:
    """
    Use Selenium to scrape prices when API doesn't work.
    Requires: pip install selenium webdriver-manager
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        print("Selenium not installed. Run: pip install selenium webdriver-manager")
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
    
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"https://www.semopx.com/market-data/market-results"
    
    # Setup headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        driver.get(url)
        
        # Wait for page to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
        )
        
        # Set the date (you'd need to interact with the date picker)
        # This is site-specific and may need adjustment
        
        time.sleep(3)  # Wait for data to load
        
        # Find the table and extract data
        table = driver.find_element(By.CSS_SELECTOR, "table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        records = []
        for row in rows[1:]:  # Skip header
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 3:
                date_val = cells[0].text.strip()
                time_val = cells[1].text.strip()
                price_val = cells[2].text.strip()
                
                try:
                    ts = pd.to_datetime(f"{date_val} {time_val}")
                    if ts.tz is None:
                        ts = ts.tz_localize("Europe/Dublin", ambiguous="NaT")
                    ts_utc = ts.tz_convert("UTC")
                    
                    price = float(price_val.replace(",", ""))
                    records.append({"ts_utc": ts_utc, "dam_eur_mwh": price})
                except Exception:
                    continue
        
        driver.quit()
        
        if records:
            df = pd.DataFrame(records)
            df = df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
            return df.reset_index(drop=True)
        
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
        
    except Exception as e:
        print(f"Selenium scrape failed: {e}")
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


if __name__ == "__main__":
    # Test the scraper
    print("Testing SEMOpx scraper...")
    
    # Try to fetch yesterday's prices
    yesterday = datetime.now().date() - timedelta(days=1)
    df = fetch_dam_for_date(yesterday)
    
    if df.empty:
        print("API method didn't work, trying alternative...")
        df = fetch_dam_alternative(yesterday)
    
    if df.empty:
        print("Trying Selenium method...")
        df = fetch_dam_selenium(yesterday)
    
    if not df.empty:
        print(f"\nSuccess! Got {len(df)} price records:")
        print(df.head(10))
    else:
        print("\nCouldn't fetch data. The SEMOpx website structure may have changed.")
        print("You may need to manually download from the Document Library.")
