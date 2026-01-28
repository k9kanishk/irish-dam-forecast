# src/data/semopx_scraper.py
"""
Fixed SEMOpx Day-Ahead Market price fetcher.

PROBLEM: The original scraper used a non-existent API endpoint:
    https://www.semopx.com/api/market-data/market-results  <-- WRONG (404)

SOLUTION: Use the official SEMOpx Report API documented at:
    https://www.semopx.com/documents/general-publications/SEMOpx-Website-Report-API.pdf

Correct API structure:
1. List reports: https://reports.semopx.com/api/v1/documents/static-reports
2. Download files: https://reports.semopx.com/documents/[ResourceName]

Key report IDs:
- EA-001: ETS Market Results (contains DAM Index Prices for ROI-DA, NI-DA)

Data sources (in order of preference):
1. SEMOpx Report API (official)
2. ENTSO-E Transparency Platform (requires free API key)
3. Local Excel/Parquet files (fallback)
"""
from __future__ import annotations
import io
import os
import time
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import requests

# === Configuration ===
CACHE_DIR = Path("data/raw/semopx_daily")
COMBINED_FILE = Path("data/raw/dam_prices_combined.parquet")

# ============================================================================
# CORRECT SEMOpx API endpoints (from official documentation)
# ============================================================================
SEMOPX_API_BASE = "https://reports.semopx.com/api/v1/documents/static-reports"
SEMOPX_DOWNLOAD_BASE = "https://reports.semopx.com/documents"

# Report IDs from SEMOpx Data Publication Guide
REPORT_IDS = {
    "ets_market_results": "EA-001",  # Contains DAM Index Prices
    "ets_bid_file": "EA-002",
    "load_forecast_annual": "BM-009",
    "load_forecast_daily": "BM-010",
    "wind_forecast": "BM-013",
    "imbalance_price": "BM-025",
}

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


# ============================================================================
# HTTP Request Utilities
# ============================================================================

def _retry_request(
    url: str,
    params: dict = None,
    max_retries: int = 3,
    timeout: int = 30
) -> requests.Response:
    """Make request with exponential backoff retry."""
    last_error = None
    for attempt in range(max_retries):
        try:
            delay = random.uniform(0.5, 1.5) * (attempt + 1)
            time.sleep(delay)
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            last_error = e
            print(f"  Attempt {attempt + 1} failed: {e.response.status_code} {e.response.reason}")
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1} failed: {e}")
    raise RuntimeError(f"All {max_retries} attempts failed: {last_error}")


# ============================================================================
# SEMOpx Report API Functions
# ============================================================================

def list_semopx_reports(
    report_id: str = "EA-001",
    start_date: str = None,
    end_date: str = None,
    page_size: int = 100
) -> List[Dict[str, Any]]:
    """
    List available reports from SEMOpx.
    
    Args:
        report_id: DPuG_ID from the Data Publication Guide (e.g., "EA-001")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        page_size: Number of results per page
        
    Returns:
        List of report metadata dictionaries
    """
    params = {
        "Group": "Market Data",
        "DPuG_ID": report_id,
        "page_size": page_size,
        "sort_by": "Date",
        "order_by": "ASC"
    }
    
    if start_date and end_date:
        params["Date"] = f">={start_date}<={end_date}"
    elif start_date:
        params["Date"] = f">={start_date}"
    elif end_date:
        params["Date"] = f"<={end_date}"
    
    all_reports = []
    page = 1
    
    while True:
        params["page"] = page
        print(f"  Fetching report list page {page}...")
        
        try:
            resp = _retry_request(SEMOPX_API_BASE, params=params)
            data = resp.json()
            
            items = data.get("items", [])
            if not items:
                break
                
            all_reports.extend(items)
            
            total_pages = data.get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1
            
        except Exception as e:
            print(f"  Error fetching report list: {e}")
            break
    
    return all_reports


def download_semopx_report(resource_name: str) -> Optional[str]:
    """
    Download a report file from SEMOpx.
    
    Args:
        resource_name: The ResourceName from the report list
        
    Returns:
        File content as string, or None if download failed
    """
    url = f"{SEMOPX_DOWNLOAD_BASE}/{resource_name}"
    
    try:
        resp = _retry_request(url)
        return resp.text
    except Exception as e:
        print(f"  Failed to download {resource_name}: {e}")
        return None


def parse_ets_xml(content: str) -> pd.DataFrame:
    """
    Parse ETS Market Results XML file.
    
    The XML contains IndexPrices for different market areas:
    - ROI-DA (Republic of Ireland Day-Ahead)
    - NI-DA (Northern Ireland Day-Ahead)
    
    Returns DataFrame with: ts_utc, dam_eur_mwh
    """
    records = []
    
    try:
        root = ET.fromstring(content)
        
        # SEMOpx XML typically uses namespaces - handle both with and without
        # Common structure: IndexPrice elements with DateTime, PriceEUR, MarketArea
        
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1].lower() if '}' in elem.tag else elem.tag.lower()
            
            if tag_name in ['indexprice', 'price', 'marketresult']:
                record = {}
                
                # Check attributes
                for attr, val in elem.attrib.items():
                    attr_lower = attr.lower()
                    if 'datetime' in attr_lower or 'time' in attr_lower:
                        record['datetime'] = val
                    elif 'priceeur' in attr_lower or 'price' in attr_lower:
                        record['price'] = val
                    elif 'marketarea' in attr_lower or 'area' in attr_lower:
                        record['area'] = val
                
                # Check child elements
                for child in elem:
                    child_tag = child.tag.split('}')[-1].lower() if '}' in child.tag else child.tag.lower()
                    child_text = (child.text or '').strip()
                    
                    if 'datetime' in child_tag or child_tag == 'time':
                        record['datetime'] = child_text
                    elif 'priceeur' in child_tag:
                        record['price'] = child_text
                    elif 'pricegbp' in child_tag and 'price' not in record:
                        record['price_gbp'] = child_text
                    elif 'marketarea' in child_tag or child_tag == 'area':
                        record['area'] = child_text
                    elif child_tag == 'volume':
                        record['volume'] = child_text
                
                # Process complete records for ROI-DA
                if record.get('datetime') and record.get('price'):
                    area = record.get('area', '').upper()
                    # Filter for ROI Day-Ahead (skip NI and intraday)
                    if 'ROI' in area and 'DA' in area:
                        try:
                            ts = pd.to_datetime(record['datetime'])
                            price = float(record['price'])
                            
                            if ts.tz is None:
                                ts = ts.tz_localize("Europe/Dublin", ambiguous="NaT", nonexistent="shift_forward")
                            ts_utc = ts.tz_convert("UTC")
                            
                            records.append({
                                "ts_utc": ts_utc,
                                "dam_eur_mwh": price
                            })
                        except Exception as e:
                            pass  # Skip malformed records
        
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
    except Exception as e:
        print(f"  Error parsing XML: {e}")
    
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
        return df.reset_index(drop=True)
    
    return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


def parse_ets_csv(content: str) -> pd.DataFrame:
    """
    Parse ETS Market Results CSV file.
    
    Returns DataFrame with: ts_utc, dam_eur_mwh
    """
    try:
        # Try different delimiters
        for delimiter in [',', ';', '\t']:
            try:
                df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        # Normalize column names
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]
        
        # Find timestamp column
        ts_col = None
        for col in df.columns:
            if any(x in col for x in ['datetime', 'timestamp', 'time', 'date', 'delivery']):
                ts_col = col
                break
        
        # Find EUR price column
        price_col = None
        for col in df.columns:
            if 'eur' in col and 'price' in col:
                price_col = col
                break
        if not price_col:
            for col in df.columns:
                if 'price' in col or 'eur' in col:
                    price_col = col
                    break
        
        # Find market area column
        area_col = None
        for col in df.columns:
            if 'area' in col or 'market' in col or 'auction' in col:
                area_col = col
                break
        
        if not ts_col or not price_col:
            print(f"  Could not identify columns. Found: {list(df.columns)}")
            return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
        
        # Filter for ROI-DA if area column exists
        if area_col:
            mask = df[area_col].astype(str).str.upper().str.contains('ROI', na=False) & \
                   df[area_col].astype(str).str.upper().str.contains('DA', na=False)
            df = df[mask]
        
        if df.empty:
            return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
        
        # Build output
        out = pd.DataFrame()
        out["ts_utc"] = pd.to_datetime(df[ts_col], errors="coerce")
        out["dam_eur_mwh"] = pd.to_numeric(df[price_col], errors="coerce")
        out = out.dropna()
        
        # Localize timestamps
        if not out.empty:
            if out["ts_utc"].dt.tz is None:
                out["ts_utc"] = out["ts_utc"].dt.tz_localize(
                    "Europe/Dublin", ambiguous="NaT", nonexistent="shift_forward"
                )
            out["ts_utc"] = out["ts_utc"].dt.tz_convert("UTC")
            out = out.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
        
        return out.reset_index(drop=True)
        
    except Exception as e:
        print(f"  Error parsing CSV: {e}")
    
    return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


def fetch_dam_from_semopx(
    start_date: datetime.date,
    end_date: datetime.date,
    cache: bool = True
) -> pd.DataFrame:
    """
    Fetch DAM prices from SEMOpx Report API.
    
    This is the PRIMARY method using the correct API.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Fetching DAM from SEMOpx: {start_str} to {end_str}")
    
    # Get report list
    reports = list_semopx_reports(
        report_id="EA-001",  # ETS Market Results
        start_date=start_str,
        end_date=end_str
    )
    
    print(f"  Found {len(reports)} reports")
    
    if not reports:
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
    
    all_frames = []
    processed_dates = set()
    
    for report in reports:
        resource_name = report.get("ResourceName", "")
        report_date = report.get("Date", "")
        report_name = report.get("ReportName", "")
        
        # Check cache
        if cache and report_date:
            cache_file = CACHE_DIR / f"semopx_{report_date}.parquet"
            if cache_file.exists() and report_date not in processed_dates:
                try:
                    df = pd.read_parquet(cache_file)
                    if not df.empty:
                        print(f"  {report_date}: loaded from cache ({len(df)} rows)")
                        all_frames.append(df)
                        processed_dates.add(report_date)
                        continue
                except Exception:
                    pass
        
        # Download
        print(f"  Downloading: {resource_name}")
        content = download_semopx_report(resource_name)
        
        if not content:
            continue
        
        # Parse (XML or CSV based on content)
        content_start = content.strip()[:100].lower()
        if '<?xml' in content_start or content_start.startswith('<'):
            df = parse_ets_xml(content)
        else:
            df = parse_ets_csv(content)
        
        if not df.empty:
            print(f"  {report_date}: extracted {len(df)} rows")
            
            if cache and report_date:
                cache_file = CACHE_DIR / f"semopx_{report_date}.parquet"
                df.to_parquet(cache_file, index=False)
            
            all_frames.append(df)
            if report_date:
                processed_dates.add(report_date)
        
        time.sleep(random.uniform(0.3, 0.7))
    
    if not all_frames:
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
    
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
    return combined.reset_index(drop=True)


# ============================================================================
# ENTSO-E Transparency Platform (Fallback)
# ============================================================================

def fetch_dam_from_entsoe(
    start_date: datetime.date,
    end_date: datetime.date,
    api_key: str = None
) -> pd.DataFrame:
    """
    Fetch DAM prices from ENTSO-E Transparency Platform.
    
    This is a FALLBACK method if SEMOpx fails.
    
    Requires:
    - Free API key from https://transparency.entsoe.eu/
    - Register and email transparency@entsoe.eu with subject "Restful API access"
    
    Or use the entsoe-py library:
        pip install entsoe-py
    """
    api_key = api_key or os.environ.get("ENTSOE_API_KEY")
    
    # Method 1: Try entsoe-py library (easier)
    try:
        from entsoe import EntsoePandasClient
        
        if not api_key:
            print("  ENTSOE_API_KEY not set")
            raise ImportError("No API key")
        
        client = EntsoePandasClient(api_key=api_key)
        
        start_ts = pd.Timestamp(start_date.strftime('%Y%m%d'), tz='Europe/Dublin')
        end_ts = pd.Timestamp((end_date + timedelta(days=1)).strftime('%Y%m%d'), tz='Europe/Dublin')
        
        # Ireland SEM uses country code 'IE_SEM' or area code '10Y1001A1001A59C'
        try:
            prices = client.query_day_ahead_prices('IE_SEM', start=start_ts, end=end_ts)
        except:
            # Fallback to area code
            prices = client.query_day_ahead_prices('10Y1001A1001A59C', start=start_ts, end=end_ts)
        
        if prices is not None and not prices.empty:
            df = prices.reset_index()
            df.columns = ['ts_utc', 'dam_eur_mwh']
            df['ts_utc'] = pd.to_datetime(df['ts_utc'], utc=True)
            print(f"  ENTSO-E: got {len(df)} rows via entsoe-py")
            return df
            
    except ImportError:
        print("  entsoe-py not installed. Install with: pip install entsoe-py")
    except Exception as e:
        print(f"  entsoe-py failed: {e}")
    
    # Method 2: Direct API call
    if not api_key:
        print("  Set ENTSOE_API_KEY environment variable or pass api_key parameter")
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
    
    url = "https://web-api.tp.entsoe.eu/api"
    domain = "10Y1001A1001A59C"  # IE_SEM bidding zone
    
    params = {
        "securityToken": api_key,
        "documentType": "A44",  # Price document
        "in_Domain": domain,
        "out_Domain": domain,
        "periodStart": start_date.strftime("%Y%m%d0000"),
        "periodEnd": (end_date + timedelta(days=1)).strftime("%Y%m%d0000"),
    }
    
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(resp.content)
        ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
        
        records = []
        for ts in root.findall(".//ns:TimeSeries", ns):
            for period in ts.findall("ns:Period", ns):
                start_elem = period.find("ns:timeInterval/ns:start", ns)
                if start_elem is None:
                    continue
                period_start = pd.to_datetime(start_elem.text)
                
                for point in period.findall("ns:Point", ns):
                    pos_elem = point.find("ns:position", ns)
                    price_elem = point.find("ns:price.amount", ns)
                    
                    if pos_elem is not None and price_elem is not None:
                        pos = int(pos_elem.text)
                        price = float(price_elem.text)
                        ts_utc = period_start + timedelta(hours=pos - 1)
                        records.append({"ts_utc": ts_utc, "dam_eur_mwh": price})
        
        if records:
            df = pd.DataFrame(records)
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
            df = df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
            print(f"  ENTSO-E: got {len(df)} rows via direct API")
            return df.reset_index(drop=True)
            
    except Exception as e:
        print(f"  ENTSO-E direct API failed: {e}")
    
    return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


# ============================================================================
# Local File Fallback
# ============================================================================

def fetch_dam_from_local_files(
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    search_paths: List[Path] = None
) -> pd.DataFrame:
    """
    Load DAM prices from local Excel/Parquet/CSV files.
    
    This is the LAST RESORT fallback.
    """
    if search_paths is None:
        search_paths = [
            Path("data/raw"),
            Path("src/data/raw"),
            Path("."),
        ]
    
    patterns = [
        "**/dam_prices*.xlsx",
        "**/dam_prices*.parquet",
        "**/dam_prices*.csv",
        "**/lookback*mkt*.xlsx",
        "**/market_results*.xlsx",
    ]
    
    all_files = []
    for base_path in search_paths:
        if not base_path.exists():
            continue
        for pattern in patterns:
            all_files.extend(base_path.glob(pattern))
    
    print(f"  Found {len(all_files)} local data files")
    
    all_frames = []
    
    for fpath in all_files:
        try:
            print(f"  Loading: {fpath}")
            
            if fpath.suffix == '.parquet':
                df = pd.read_parquet(fpath)
            elif fpath.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(fpath, engine='openpyxl')
            elif fpath.suffix == '.csv':
                df = pd.read_csv(fpath)
            else:
                continue
            
            # Normalize columns
            cols = {str(c).lower().strip(): c for c in df.columns}
            
            # Find timestamp
            ts_col = None
            for key in ['ts_utc', 'timestamp', 'datetime', 'date', 'time']:
                if key in cols:
                    ts_col = cols[key]
                    break
            
            # Find price
            price_col = None
            for key in ['dam_eur_mwh', 'price_eur', 'eur/mwh', 'price', 'value']:
                if key in cols:
                    price_col = cols[key]
                    break
            
            if ts_col and price_col:
                out = pd.DataFrame({
                    "ts_utc": pd.to_datetime(df[ts_col], errors="coerce"),
                    "dam_eur_mwh": pd.to_numeric(df[price_col], errors="coerce")
                }).dropna()
                
                if not out.empty:
                    # Handle timezone
                    if out["ts_utc"].dt.tz is None:
                        try:
                            out["ts_utc"] = out["ts_utc"].dt.tz_localize("UTC")
                        except:
                            out["ts_utc"] = out["ts_utc"].dt.tz_localize("Europe/Dublin").dt.tz_convert("UTC")
                    else:
                        out["ts_utc"] = out["ts_utc"].dt.tz_convert("UTC")
                    
                    all_frames.append(out)
                    print(f"    Loaded {len(out)} rows")
                    
        except Exception as e:
            print(f"    Error loading {fpath}: {e}")
    
    if not all_frames:
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])
    
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
    
    # Filter by date range if specified
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        combined = combined[combined["ts_utc"] >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC") + timedelta(days=1)
        combined = combined[combined["ts_utc"] < end_ts]
    
    return combined.reset_index(drop=True)


# ============================================================================
# Main Interface Functions
# ============================================================================

def fetch_dam_prices(
    days: int = 90,
    force_refresh: bool = False,
    entsoe_api_key: str = None
) -> pd.DataFrame:
    """
    Fetch DAM prices using multiple sources with fallback.
    
    Priority:
    1. SEMOpx Report API (official source)
    2. ENTSO-E Transparency Platform (reliable fallback)
    3. Local files (last resort)
    
    Args:
        days: Number of days of history to fetch
        force_refresh: If True, ignore cache
        entsoe_api_key: Optional ENTSO-E API key
        
    Returns:
        DataFrame with columns: ts_utc, dam_eur_mwh
    """
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    print(f"Fetching DAM prices for {days} days ({start_date} to {end_date})")
    print("=" * 60)
    
    # Method 1: SEMOpx Report API
    print("\n[1/3] Trying SEMOpx Report API...")
    try:
        df = fetch_dam_from_semopx(start_date, end_date, cache=not force_refresh)
        if not df.empty and len(df) >= 24:
            print(f"  SUCCESS: Got {len(df)} rows from SEMOpx")
            return df
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Method 2: ENTSO-E
    print("\n[2/3] Trying ENTSO-E Transparency Platform...")
    try:
        df = fetch_dam_from_entsoe(start_date, end_date, api_key=entsoe_api_key)
        if not df.empty and len(df) >= 24:
            print(f"  SUCCESS: Got {len(df)} rows from ENTSO-E")
            return df
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Method 3: Local files
    print("\n[3/3] Trying local files...")
    try:
        df = fetch_dam_from_local_files(start_date, end_date)
        if not df.empty:
            print(f"  SUCCESS: Got {len(df)} rows from local files")
            return df
    except Exception as e:
        print(f"  FAILED: {e}")
    
    print("\nAll methods failed!")
    return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


def fetch_recent_dam(days: int = 30, force_refresh: bool = False) -> pd.DataFrame:
    """
    Convenience function: fetch last N days of DAM prices.
    
    This is the function called by daily_update.py
    """
    df = fetch_dam_prices(days=days, force_refresh=force_refresh)
    
    # Save combined file
    if not df.empty:
        COMBINED_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(COMBINED_FILE, index=False)
        print(f"\nSaved {len(df)} rows to {COMBINED_FILE}")
    
    return df


def load_combined_dam() -> pd.DataFrame:
    """Load the combined DAM prices file."""
    if COMBINED_FILE.exists():
        df = pd.read_parquet(COMBINED_FILE)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        return df
    return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


# ============================================================================
# Testing & Diagnostics
# ============================================================================

def test_api_connectivity():
    """Test connectivity to various data sources."""
    print("Testing API connectivity...")
    print("=" * 60)
    
    # Test SEMOpx
    print("\n[SEMOpx Report API]")
    try:
        resp = requests.get(
            SEMOPX_API_BASE,
            params={"Group": "Market Data", "page_size": 1},
            headers=HEADERS,
            timeout=10
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Total reports available: {data.get('totalItems', 'N/A')}")
        else:
            print(f"  Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test ENTSO-E
    print("\n[ENTSO-E API]")
    api_key = os.environ.get("ENTSOE_API_KEY")
    if api_key:
        try:
            url = "https://web-api.tp.entsoe.eu/api"
            params = {
                "securityToken": api_key,
                "documentType": "A44",
                "in_Domain": "10Y1001A1001A59C",
                "out_Domain": "10Y1001A1001A59C",
                "periodStart": "202501010000",
                "periodEnd": "202501020000",
            }
            resp = requests.get(url, params=params, timeout=10)
            print(f"  Status: {resp.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  ENTSOE_API_KEY not set")
        print("  To get a key: https://transparency.entsoe.eu/")
        print("  Email: transparency@entsoe.eu with subject 'Restful API access'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEMOpx DAM Price Fetcher (Fixed)")
    parser.add_argument("--days", type=int, default=7, help="Days of history to fetch")
    parser.add_argument("--test", action="store_true", help="Test API connectivity")
    parser.add_argument("--force", action="store_true", help="Force refresh (ignore cache)")
    
    args = parser.parse_args()
    
    if args.test:
        test_api_connectivity()
    else:
        df = fetch_recent_dam(days=args.days, force_refresh=args.force)
        
        if not df.empty:
            print("\n" + "=" * 60)
            print("Results:")
            print(f"  Records: {len(df)}")
            print(f"  Date range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
            print(f"  Price range: €{df['dam_eur_mwh'].min():.2f} - €{df['dam_eur_mwh'].max():.2f}/MWh")
            print(f"  Mean price: €{df['dam_eur_mwh'].mean():.2f}/MWh")
            print("\nSample data:")
            print(df.head(10))
        else:
            print("\nNo data retrieved.")
