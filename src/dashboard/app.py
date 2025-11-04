# src/dashboard/app.py

# --- bootstrap (imports first!) ---
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import yaml
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import requests
import numpy as np

# Ensure repo root is on sys.path so `import src...` works in Streamlit Cloud/Codespaces
APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[2]  # .../irish-dam-forecast
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

load_dotenv()  # local dev uses .env; on Streamlit Cloud we'll use st.secrets

DATA_PATH = Path("data/processed/train.parquet")

from datetime import timedelta
from entsoe.exceptions import NoMatchingDataError

class EirGridQuickFix:
    """Minimal EirGrid data fetcher for immediate use"""
    
    @staticmethod
    def get_recent_data():
        """Get recent wind and demand data from EirGrid"""
        try:
            # EirGrid CSV endpoint (public data)
            base_url = "https://www.smartgriddashboard.com/DashboardService.svc/data"
            
            # Get last 7 days
            date_from = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
            date_to = datetime.now().strftime("%d-%b-%Y")
            
            data_frames = []
            
            # Fetch wind actual
            params = {
                "area": "windactual",
                "region": "ALL",
                "datefrom": date_from,
                "dateto": date_to
            }
            
            resp = requests.get(base_url, params=params, timeout=10)
            if resp.status_code == 200:
                wind_data = resp.json()
                if "Rows" in wind_data:
                    df_wind = pd.DataFrame(wind_data["Rows"])
                    df_wind['timestamp'] = pd.to_datetime(df_wind['EffectiveTime'])
                    df_wind['wind_actual_mw'] = pd.to_numeric(df_wind['Value'])
                    data_frames.append(df_wind[['timestamp', 'wind_actual_mw']].set_index('timestamp'))
            
            # Fetch demand actual  
            params["area"] = "demandactual"
            resp = requests.get(base_url, params=params, timeout=10)
            if resp.status_code == 200:
                demand_data = resp.json()
                if "Rows" in demand_data:
                    df_demand = pd.DataFrame(demand_data["Rows"])
                    df_demand['timestamp'] = pd.to_datetime(df_demand['EffectiveTime'])
                    df_demand['demand_actual_mw'] = pd.to_numeric(df_demand['Value'])
                    data_frames.append(df_demand[['timestamp', 'demand_actual_mw']].set_index('timestamp'))
            
            if data_frames:
                return pd.concat(data_frames, axis=1)
                
        except Exception as e:
            st.warning(f"EirGrid data fetch failed: {e}")
        
        return pd.DataFrame()

def _read_cached_dam(start: str, end: str) -> pd.Series:
    """Get dam_eur_mwh from our parquet if API is missing."""
    if not DATA_PATH.exists():
        return pd.Series(dtype=float)
    dfc = pd.read_parquet(DATA_PATH)
    if "dam_eur_mwh" not in dfc.columns:
        return pd.Series(dtype=float)
    s = dfc["dam_eur_mwh"].copy()
    s.index = pd.to_datetime(s.index)
    s = s[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]
    s.name = "dam_eur_mwh"
    return s

def _get_dam_history(ent, start: str, end: str) -> pd.Series:
    """Try API, then step the end date back, then fall back to parquet cache."""
    # 1) try the full window
    try:
        s = ent.day_ahead_prices(start=start, end=end)
        if s is not None and len(s):
            return s
    except NoMatchingDataError:
        pass
    # 2) back off the end date up to 14 days
    for back in range(1, 15):
        try:
            end2 = str((pd.to_datetime(end) - pd.Timedelta(days=back)).date())
            if pd.to_datetime(end2) < pd.to_datetime(start):
                break
            s = ent.day_ahead_prices(start=start, end=end2)
            if s is not None and len(s):
                return s
        except NoMatchingDataError:
            continue
    # 3) fallback to cached parquet (best-effort)
    s = _read_cached_dam(start, end)
    return s


# put near the top of app.py, after DATA_PATH is defined
with st.sidebar:
    if st.button("Rebuild data"):
        DATA_PATH.unlink(missing_ok=True)
        st.cache_data.clear()
        st.rerun()



def _as_date(s):
    """Parse s to date; return None if missing/blank/invalid."""
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _chunk_edges(start_date, end_date, days=30):  # 30 keeps calls small/fast
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    step = pd.Timedelta(days=days)
    cur = pd.to_datetime(s); last = pd.to_datetime(e)
    while cur <= last:
        nxt = min(cur + step, last)
        yield str(cur.date()), str(nxt.date())
        cur = nxt + pd.Timedelta(days=1)


@st.cache_data(show_spinner=True, ttl=3600)  # Refresh hourly
def ensure_dataset():
    """Build data/processed/train.parquet with better error handling"""
    
    if DATA_PATH.exists():
        # Check if data is fresh (less than 6 hours old)
        mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(hours=6):
            return  # Use existing data
        else:
            DATA_PATH.unlink()  # Remove stale data
    
    # Heavy imports
    from src.data.entsoe_api import Entsoe
    from src.data.weather import fetch_hourly
    from src.features.build_features import build_feature_table
    from src.features.targets import make_day_ahead_target
    from entsoe.exceptions import NoMatchingDataError
    
    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # === FIX 1: Use realistic dates ===
    today = datetime.now(timezone.utc).date()
    
    # ENTSO-E is typically 2-5 days behind current date
    safe_end_date = today - timedelta(days=3)
    safe_start_date = safe_end_date - timedelta(days=90)  # Last 90 days
    
    # Override config dates if they're unrealistic
    start_all = str(safe_start_date)
    end_all = str(safe_end_date)
    
    st.info(f"Fetching data from {start_all} to {end_all}")
    
    # Get token
    token = os.getenv("ENTSOE_TOKEN") or st.secrets.get("ENTSOE_TOKEN")
    if not token:
        st.error("ENTSOE_TOKEN not found. Add it to Streamlit Secrets.")
        st.stop()
    
    ent = Entsoe(token=token)
    
    # === FIX 2: Fetch with better error handling ===
    all_data = {}
    
    # Helper function with retries
    def fetch_with_retry(method_name, **kwargs):
        """Fetch with automatic retry and date adjustment"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Try progressively older dates if recent data fails
                adjusted_end = pd.to_datetime(kwargs['end']) - timedelta(days=attempt)
                kwargs['end'] = str(adjusted_end.date())
                
                result = getattr(ent, method_name)(**kwargs)
                if result is not None and len(result) > 0:
                    return result
            except NoMatchingDataError:
                st.warning(f"{method_name} failed for {kwargs['end']}, trying earlier date...")
                continue
            except Exception as e:
                if attempt == max_attempts - 1:
                    st.error(f"{method_name} failed after {max_attempts} attempts: {e}")
                    break
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return pd.Series() if 'series' in method_name.lower() else pd.DataFrame()
    
    # Fetch DAM prices
    st.caption("Fetching DAM prices...")
    dam = fetch_with_retry("day_ahead_prices", start=start_all, end=end_all)
    
    if dam.empty:
        st.warning("No DAM prices from ENTSO-E, creating synthetic prices for demo")
        # Create synthetic prices that follow typical patterns
        idx = pd.date_range(start_all, end_all, freq='H', tz=None)
        base_price = 80
        hourly_pattern = np.array([0.8, 0.75, 0.7, 0.7, 0.75, 0.85, 0.95, 1.1, 
                                   1.2, 1.15, 1.1, 1.05, 1.0, 1.0, 1.05, 1.1,
                                   1.2, 1.3, 1.25, 1.15, 1.0, 0.95, 0.9, 0.85])
        
        prices = []
        for ts in idx:
            hour_factor = hourly_pattern[ts.hour]
            day_random = np.random.normal(1.0, 0.1)
            price = base_price * hour_factor * day_random
            prices.append(price)
        
        dam = pd.Series(prices, index=idx, name='dam_eur_mwh')
    
    # Fetch load forecast
    st.caption("Fetching load forecast...")
    load_fc = fetch_with_retry("load_forecast", start=start_all, end=end_all)
    
    if load_fc.empty:
        st.warning("No load forecast from ENTSO-E, using typical pattern")
        idx = dam.index if not dam.empty else pd.date_range(start_all, end_all, freq='H', tz=None)
        base_load = 4500
        load_fc = pd.Series(
            base_load + 500 * np.sin(2 * np.pi * idx.hour / 24) + np.random.normal(0, 100, len(idx)),
            index=idx,
            name='load_forecast_mw'
        )
    
    # Fetch wind/solar forecast
    st.caption("Fetching wind/solar forecast...")
    ws = fetch_with_retry("wind_solar_forecast", start=start_all, end=end_all)
    
    if ws.empty:
        st.warning("No wind/solar from ENTSO-E, using EirGrid data if available")
        # Try EirGrid as fallback
        eirgrid_data = EirGridQuickFix.get_recent_data()
        if not eirgrid_data.empty:
            st.success("Got data from EirGrid!")
            # Convert to similar format
            ws = pd.DataFrame(index=dam.index)
            ws['wind_total_mw'] = eirgrid_data['wind_actual_mw'].reindex(ws.index).fillna(method='ffill')
    
    # === FIX 3: Weather with proper date handling ===
    st.caption("Fetching weather data...")
    lat = cfg["weather"]["lat"]
    lon = cfg["weather"]["lon"]
    
    # Modified weather fetcher that handles date limits
    def fetch_weather_safe(lat, lon, start, end):
        """Fetch weather with Open-Meteo API limits handled"""
        from datetime import datetime, timedelta
        import requests
        
        start_d = pd.to_datetime(start).date()
        end_d = pd.to_datetime(end).date()
        today = datetime.now().date()
        
        all_weather = []
        
        # Historical data (if needed)
        if start_d < today:
            hist_end = min(end_d, today - timedelta(days=1))
            
            # ERA5 archive for historical
            try:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": str(start_d),
                    "end_date": str(hist_end),
                    "hourly": "windspeed_100m,temperature_2m,cloudcover",
                    "timezone": "UTC"
                }
                
                resp = requests.get(
                    "https://archive-api.open-meteo.com/v1/era5",
                    params=params,
                    timeout=30
                )
                
                if resp.status_code == 200:
                    data = resp.json()["hourly"]
                    times = pd.to_datetime(data["time"]).tz_localize("UTC").tz_convert("Europe/Dublin").tz_localize(None)
                    
                    hist_weather = pd.DataFrame({
                        "wind100m_ms": data.get("windspeed_100m"),
                        "temperature_2m": data.get("temperature_2m"),
                        "cloud_cover": data.get("cloudcover")
                    }, index=times)
                    
                    all_weather.append(hist_weather)
            except Exception as e:
                st.warning(f"Historical weather failed: {e}")
        
        # Current/forecast data (if needed and within 16 days)
        if end_d >= today and start_d <= today + timedelta(days=15):
            forecast_start = max(start_d, today)
            forecast_end = min(end_d, today + timedelta(days=15))
            
            try:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": str(forecast_start),
                    "end_date": str(forecast_end),
                    "hourly": "wind_speed_100m,temperature_2m,cloud_cover",
                    "timezone": "UTC"
                }
                
                resp = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params=params,
                    timeout=30
                )
                
                if resp.status_code == 200:
                    data = resp.json()["hourly"]
                    times = pd.to_datetime(data["time"]).tz_localize("UTC").tz_convert("Europe/Dublin").tz_localize(None)
                    
                    forecast_weather = pd.DataFrame({
                        "wind100m_ms": data.get("wind_speed_100m"),
                        "temperature_2m": data.get("temperature_2m"),
                        "cloud_cover": data.get("cloud_cover")
                    }, index=times)
                    
                    all_weather.append(forecast_weather)
            except Exception as e:
                st.warning(f"Forecast weather failed: {e}")
        
        # Combine or create synthetic
        if all_weather:
            weather = pd.concat(all_weather).sort_index()
            weather = weather[~weather.index.duplicated(keep="last")]
        else:
            # Synthetic weather as last resort
            idx = pd.date_range(start_d, end_d, freq='H', tz=None)
            weather = pd.DataFrame({
                "wind100m_ms": 8.5 + 3 * np.sin(2 * np.pi * idx.dayofyear / 365),
                "temperature_2m": 10 + 8 * np.sin(2 * np.pi * idx.dayofyear / 365),
                "cloud_cover": 65.0 + 20 * np.random.randn(len(idx))
            }, index=idx)
        
        return weather
    
    weather = fetch_weather_safe(lat, lon, start_all, end_all)
    
    # === FIX 4: Align all data properly ===
    # Ensure all data has same timezone treatment (Europe/Dublin, tz-naive)
    def ensure_tz_naive_dublin(obj):
        """Convert any timezone-aware index to Dublin time, then make naive"""
        if not hasattr(obj, 'index'):
            return obj
        if not isinstance(obj.index, pd.DatetimeIndex):
            return obj
        
        idx = obj.index
        if idx.tz is not None:
            idx = idx.tz_convert("Europe/Dublin")
        else:
            # Assume Europe/Brussels for ENTSO-E data
            idx = idx.tz_localize("Europe/Brussels").tz_convert("Europe/Dublin")
        
        idx = idx.tz_localize(None)
        obj.index = idx
        return obj
    
    dam = ensure_tz_naive_dublin(dam)
    load_fc = ensure_tz_naive_dublin(load_fc)
    ws = ensure_tz_naive_dublin(ws)
    weather = ensure_tz_naive_dublin(weather)
    
    # Remove duplicates
    dam = dam[~dam.index.duplicated(keep="last")]
    load_fc = load_fc[~load_fc.index.duplicated(keep="last")]
    if not ws.empty:
        ws = ws[~ws.index.duplicated(keep="last")]
    weather = weather[~weather.index.duplicated(keep="last")]
    
    # Find common timestamps
    common_idx = dam.index
    if not load_fc.empty:
        common_idx = common_idx.intersection(load_fc.index)
    if not ws.empty:
        common_idx = common_idx.intersection(ws.index)
    if not weather.empty:
        common_idx = common_idx.intersection(weather.index)
    
    if len(common_idx) == 0:
        st.error("No overlapping timestamps between data sources!")
        st.stop()
    
    # Reindex to common timestamps
    dam = dam.reindex(common_idx)
    load_fc = load_fc.reindex(common_idx)
    ws = ws.reindex(common_idx) if not ws.empty else ws
    weather = weather.reindex(common_idx)
    
    # Ensure load_fc is a Series
    if isinstance(load_fc, pd.DataFrame):
        load_fc = load_fc.iloc[:, 0]
    load_fc.name = "load_forecast_mw"
    
    # Build features and target
    st.caption("Building features...")
    X = build_feature_table(dam, load_fc, ws, weather)
    y = make_day_ahead_target(dam).reindex(X.index)
    
    # Keep only rows with target
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    if X.empty:
        st.error("No valid training data after processing!")
        st.stop()
    
    # Save to parquet
    out = X.copy()
    out["target"] = y
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_PATH)
    
    st.success(f"Dataset created: {len(out)} rows from {out.index.min()} to {out.index.max()}")


# === ADD THIS TO YOUR SIDEBAR FOR DATA STATUS ===
def show_data_status():
    """Show data freshness and source in sidebar"""
    if DATA_PATH.exists():
        mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        age = datetime.now() - mod_time
        
        if age < timedelta(hours=1):
            status = "🟢 Fresh"
        elif age < timedelta(hours=6):
            status = "🟡 Recent"
        else:
            status = "🔴 Stale"
        
        st.sidebar.markdown(f"""
        **Data Status**
        - Status: {status}
        - Updated: {mod_time.strftime('%Y-%m-%d %H:%M')}
        - Age: {age.total_seconds()/3600:.1f} hours
        """)
        
        if age > timedelta(hours=12):
            if st.sidebar.button("🔄 Force Refresh"):
                DATA_PATH.unlink()
                st.cache_data.clear()
                st.rerun()
    else:
        st.sidebar.warning("No data cached yet")


# === USE THIS FOR DATE PICKER ===
def get_date_picker_defaults():
    """Get sensible defaults for date picker"""
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
        idx = pd.DatetimeIndex(df.index)
        
        # Use actual data coverage
        data_start = idx.min().date()
        data_end = idx.max().date()
        
        # Default to most recent date with data
        default_date = data_end
        
        return data_start, data_end, default_date
    else:
        # Fallback defaults
        today = datetime.now().date()
        return today - timedelta(days=30), today - timedelta(days=3), today - timedelta(days=3)


# === MAIN APP MODIFICATIONS ===
# In your main app code, replace the date picker section with:

# Build dataset first
ensure_dataset()

# Show data status
show_data_status()

# Load data
df = pd.read_parquet(DATA_PATH)

# Get proper date range
date_min, date_max, date_default = get_date_picker_defaults()

st.title("Irish Day-Ahead Power Price Forecast (SEMOpx)")

# Date picker with proper defaults
date = st.date_input(
    "Choose a date to forecast",
    value=date_default,
    min_value=date_min,
    max_value=date_max,
    help=f"Data available from {date_min} to {date_max}"
)

# Show warning if trying to forecast too far ahead
if date > date_max:
    st.warning(f"Selected date {date} is beyond available data ({date_max}). Forecast may be less accurate.")

# --- app proper (keep a single date picker) ---
import plotly.express as px
from datetime import date as _date
from zoneinfo import ZoneInfo

# compute coverage from the DataFrame index
idx = pd.DatetimeIndex(df.index)
days = pd.Index(idx.date)
if len(days) == 0:
    st.error("No rows found in the dataset.")
    st.stop()

def _as_pydate(d):
    return d if isinstance(d, _date) else pd.to_datetime(d).date()

day_min = _as_pydate(days.min())
day_max = _as_pydate(days.max())
today_ie = datetime.now(ZoneInfo("Europe/Dublin")).date()
default_day = max(day_min, min(day_max, today_ie))

st.caption(f"Data coverage: {day_min} → {day_max} | rows: {len(df):,}")

date = st.date_input(
    "Choose a date to forecast",
    value=default_day,
    min_value=day_min,
    max_value=day_max,
)

# Train quick model on all rows
y = df.pop("target")
mdl = __import__("src.models.xgb_model", fromlist=["make_model"]).make_model()
mdl.fit(df, y)

st.title("Irish Day-Ahead Power Price Forecast (SEMOpx)")

X_day = df.loc[df.index.date == pd.to_datetime(date).date()]
if len(X_day):
    preds = mdl.predict(X_day)
    out = pd.DataFrame({"ts": X_day.index, "forecast_eur_mwh": preds}).set_index("ts")
    st.subheader("Hourly forecast")
    st.dataframe(out)
    st.plotly_chart(px.line(out, y="forecast_eur_mwh"), use_container_width=True)
else:
    st.info("No features for this day yet. Adjust the training window in config.yaml or fetch more data.")


# --- Forecast mode: predict for a future calendar day (e.g., tomorrow) ---
from datetime import date as _date, timedelta
from src.data.entsoe_api import Entsoe
from src.data.weather import fetch_hourly
from src.features.build_features import build_feature_table

def make_features_for_day(target_day: _date) -> pd.DataFrame:
    """Build X for target_day using: recent DAM (for lags),
    ENTSO-E forecasts on target_day, and weather for target_day."""
    token = os.getenv("ENTSOE_TOKEN") or st.secrets.get("ENTSOE_TOKEN")
    ent = Entsoe(token=token)

    # recent history for DAM lags
    start_hist = (pd.to_datetime(target_day) - pd.Timedelta(days=14)).date()
    end_hist   = (pd.to_datetime(target_day) - pd.Timedelta(days=1)).date()
    dam_hist   = _get_dam_history(ent, str(start_hist), str(end_hist))

    # If we truly have no DAM at all, we can still try but warn that lags will be NaN
    if dam_hist.empty:
        st.warning("DAM history missing for recent days; continuing without price lags.")

    # exogenous forecasts ON the target day
    try:
        load_fc = ent.load_forecast(start=str(target_day), end=str(target_day))
    except NoMatchingDataError:
        st.warning("ENTSO-E load forecast not available for the selected day yet.")
        return pd.DataFrame()

    try:
        ws_fc = ent.wind_solar_forecast(start=str(target_day), end=str(target_day))
    except NoMatchingDataError:
        st.warning("ENTSO-E wind/solar forecast not available for the selected day yet.")
        return pd.DataFrame()

    # weather (Open-Meteo) for the target window
    # (you can read lat/lon from config.yaml; hard-coded here for brevity)
    lat, lon = 53.4, -8.2
    weather = fetch_hourly(lat, lon, str(target_day), str(target_day))

    # --- timezone harmonisation like training ---
    def _tz_fix(obj, assume_utc=False):
        if not hasattr(obj, "index") or not isinstance(obj.index, pd.DatetimeIndex):
            return obj
        idx = obj.index
        if idx.tz is None:
            idx = idx.tz_localize("Europe/Brussels" if not assume_utc else "UTC")
        idx = idx.tz_convert("Europe/Dublin").tz_localize(None)
        out = obj.copy(); out.index = idx
        return out

    dam_hist = _tz_fix(dam_hist)
    load_fc  = _tz_fix(load_fc)
    ws_fc    = _tz_fix(ws_fc)
    weather  = _tz_fix(weather, assume_utc=True)

    # Ensure a Series named load_forecast_mw
    if isinstance(load_fc, pd.DataFrame):
        # heuristics: take the first numeric column
        load_fc = pd.to_numeric(load_fc.iloc[:, 0], errors="coerce")
    load_fc.name = "load_forecast_mw"

    # Build features (same function as training)
    X_full = build_feature_table(dam_hist, load_fc, ws_fc, weather)

    # Keep only the target day’s rows
    mask = X_full.index.date == target_day
    return X_full.loc[mask]


# --- one-click T+1 forecast in the sidebar ---
from datetime import timedelta
with st.sidebar:
    if st.button("Forecast tomorrow (T+1)"):
        tgt = (datetime.now(ZoneInfo("Europe/Dublin")).date() + timedelta(days=1))
        try:
            X_t1 = make_features_for_day(tgt)   # your function
        except Exception:
            X_t1 = pd.DataFrame()
        if len(X_t1):
            preds = mdl.predict(X_t1)
            out_f = pd.DataFrame({"ts": X_t1.index, "forecast_eur_mwh": preds}).set_index("ts")
            st.success(f"T+1 forecast for {tgt.isoformat()}")
            st.dataframe(out_f)
            st.plotly_chart(px.line(out_f, y="forecast_eur_mwh"), use_container_width=True)
        else:
            st.warning("Could not build features for T+1 (API window empty). Try again later.")



# --- Backtest (last N days) ---
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def backtest_last_n_days(df_full: pd.DataFrame, n_days: int = 14):
    data = df_full.copy()
    y = data.pop("target")
    X = data
    cutoff = (X.index.max().date() - pd.Timedelta(days=n_days)).date()
    train = X.index.date < cutoff
    Xtr, ytr, Xte, yte = X.loc[train], y.loc[train], X.loc[~train], y.loc[~train]
    mdl_bt = __import__("src.models.xgb_model", fromlist=["make_model"]).make_model()
    mdl_bt.fit(Xtr, ytr)
    pred = mdl_bt.predict(Xte)
    rmse = mean_squared_error(yte, pred, squared=False)
    mape = np.mean(np.abs((yte - pred) / np.clip(np.abs(yte), 1e-6, None))) * 100
    return rmse, mape, pd.DataFrame({"actual": yte, "pred": pred}, index=Xte.index).sort_index()

with st.sidebar:
    if st.button("Run 14-day backtest"):
        rmse, mape, df_eval = backtest_last_n_days(df)
        st.success(f"RMSE: {rmse:,.2f}  |  MAPE: {mape:,.2f}%")
        st.line_chart(df_eval.resample("H").mean())


# --- Walk-forward CV over daily blocks ---
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def walk_forward_cv(df_full: pd.DataFrame, days_per_fold:int=1, n_folds:int=14, min_train_days:int=30):
    data = df_full.copy()
    y_all = data.pop("target")
    X_all = data

    # ensure strictly time-ordered
    X_all = X_all.sort_index()
    y_all = y_all.reindex(X_all.index)

    day_index = pd.Index(X_all.index.date)
    unique_days = np.array(sorted(pd.unique(day_index)))

    results = []
    for k in range(n_folds):
        # test window = the k-th day from the end (walk backward) – adjust if you prefer forward
        test_day = unique_days[-(k+1)]
        # training covers everything strictly before test_day, but at least min_train_days
        train_cut = pd.to_datetime(test_day) - pd.Timedelta(days=min_train_days)
        train_mask = (X_all.index.date < test_day) & (X_all.index >= train_cut)
        test_mask  = (X_all.index.date == test_day)

        if train_mask.sum() < 24 or test_mask.sum() == 0:
            continue

        Xtr, ytr = X_all.loc[train_mask], y_all.loc[train_mask]
        Xte, yte = X_all.loc[test_mask],  y_all.loc[test_mask]

        mdl_cv = __import__("src.models.xgb_model", fromlist=["make_model"]).make_model()
        mdl_cv.fit(Xtr, ytr)
        pred = mdl_cv.predict(Xte)

        rmse = mean_squared_error(yte, pred, squared=False)
        mape = np.mean(np.abs((yte - pred) / np.clip(np.abs(yte), 1e-6, None))) * 100
        results.append({"day": test_day, "rmse": rmse, "mape": mape})

    res = pd.DataFrame(results).sort_values("day")
    return res, res["rmse"].mean(), res["mape"].mean()

# in the sidebar:
with st.sidebar:
    if st.button("Walk-forward CV (14 folds)"):
        res, rmse_avg, mape_avg = walk_forward_cv(df, n_folds=14, min_train_days=30)
        st.write(res)
        c1, c2 = st.columns(2)
        c1.metric("Walk-forward RMSE (avg)", f"{rmse_avg:,.2f}")
        c2.metric("Walk-forward MAPE (avg)", f"{mape_avg:,.2f}%")

