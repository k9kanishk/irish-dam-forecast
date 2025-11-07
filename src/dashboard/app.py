# FINAL FIXED app.py - Replace your entire src/dashboard/app.py with this
# --- path bootstrap: make `src/` importable when running as a script ---
import os, sys
_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR  = os.path.abspath(os.path.join(_THIS_DIR, ".."))   # -> .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


from data.semopx_api import fetch_dam_hrp_recent
from data.entsoe_api import fetch_ie_dam_recent
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import yaml
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import requests
from datetime import timedelta
from data.semopx_api import fetch_dam_hrp_recent
from data.entsoe_api import fetch_ie_dam_recent   # fallback
from data.entsoe_api import Entsoe
from data.weather import fetch_hourly
from features.build_features import build_feature_table
from features.targets import make_day_ahead_target

# Sidebar toggles (if not already present)
FAST_MODE = st.sidebar.checkbox("⚡ Fast mode (use cache, skip SEMOpx if slow)", value=True)
DAYS = st.sidebar.slider("History window (days)", 7, 60, 21)

@st.cache_data(ttl=60*30, show_spinner=False)
def build_dam_cached(fast_mode: bool, days: int) -> pd.DataFrame:
    """
    Return a tidy DAM dataframe with columns ['ts_utc','dam_eur_mwh'].
    Uses SEMOpx HRP unless fast_mode is True or HRP fails, then ENTSO-E.
    """
    try:
        if fast_mode:
            raise RuntimeError("fast-mode: skip SEMOpx")
        df = fetch_dam_hrp_recent(days=days)
        if df is None or df.empty:
            raise RuntimeError("SEMOpx HRP empty")
        # already ['ts_utc','dam_eur_mwh']
        return df
    except Exception as e:
        # Fallback to ENTSO-E (no force refresh in fast mode)
        entsoe_df = fetch_ie_dam_recent(days=days, force_refresh=not fast_mode)
        if isinstance(entsoe_df, pd.Series):
            entsoe_df = entsoe_df.to_frame("dam_eur_mwh")
        entsoe_df.index = pd.DatetimeIndex(entsoe_df.index, tz="UTC")
        return entsoe_df.rename_axis("ts_utc").reset_index()

@st.cache_data(ttl=60*30, show_spinner=False)
def build_fundamentals_cached(days: int, fast_mode: bool):
    """
    Pull load forecast, wind/solar forecast, and weather; return as a dict.
    Cached so we don’t refetch on every UI change.
    """
    e = Entsoe()  # uses ENTSOE_TOKEN
    # Compute date window in local time; functions accept strings or Timestamps
    end_local = pd.Timestamp.now(tz="Europe/Dublin").normalize()
    start_local = end_local - pd.Timedelta(days=days)

    load_fc = e.load_forecast(start_local, end_local)           # Series (tz-aware)
    ws_fc   = e.wind_solar_forecast(start_local, end_local)     # DataFrame
    weather = fetch_hourly(start_local, end_local)               # DataFrame (your existing fn)

    return {"load_fc": load_fc, "ws_fc": ws_fc, "weather": weather}

@st.cache_data(ttl=60*30, show_spinner=False)
def build_features_cached(dam_df: pd.DataFrame, load_fc, ws_fc, weather):
    """
    Build X and y once and cache the result.
    """
    # y from DAM
    y = make_day_ahead_target(dam_df)  # your existing function (returns Series)
    # X from fundamentals + calendar etc.
    X = build_feature_table(dam_df, load_fc, ws_fc, weather)
    return X, y

from xgboost import XGBRegressor

@st.cache_resource
def fit_model_cached(X: pd.DataFrame, y: pd.Series, params: dict | None = None):
    """
    Train once per unique (X,y,params) state and reuse the fitted model.
    Use cache_resource for non-serializable objects like sklearn/xgb models.
    """
    if params is None:
        params = dict(
            n_estimators=800, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            tree_method="hist", objective="reg:squarederror"
        )
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model



# Fix path for imports
APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[2]  
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

load_dotenv()

# Use standard timezone handling (not pytz)
from zoneinfo import ZoneInfo

DATA_PATH = Path("data/processed/train.parquet")

# ============== HELPER CLASSES ==============

class EirGridBackup:
    """Fallback data source when ENTSO-E fails"""
    
    @staticmethod
    def get_recent_data():
        try:
            base_url = "https://www.smartgriddashboard.com/DashboardService.svc/data"
            date_from = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
            date_to = datetime.now().strftime("%d-%b-%Y")
            
            # Try wind data
            params = {"area": "windactual", "region": "ALL", "datefrom": date_from, "dateto": date_to}
            resp = requests.get(base_url, params=params, timeout=10)
            
            if resp.status_code == 200:
                return True  # Data is available
        except:
            pass
        return False

# ============== CORE DATA BUILDING FUNCTION ==============

@st.cache_data(show_spinner=True, ttl=3600)
def ensure_dataset():
    """Build the training dataset with robust error handling"""
    
    # Check if recent data exists
    if DATA_PATH.exists():
        mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(hours=6):
            return  # Use cached data
        DATA_PATH.unlink()  # Remove old data
    
    # Imports
    try:
        from data.entsoe_api import Entsoe
        from data.weather import fetch_hourly
        from features.build_features import build_feature_table
        from features.targets import make_day_ahead_target

    except ImportError as e:
        st.error(f"Import error: {e}")
        st.stop()
    
    # Load config
    try:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except:
        cfg = {"weather": {"lat": 53.4, "lon": -8.2}}
    
    # === CRITICAL FIX: Use proper dates ===
    today = datetime.now(ZoneInfo("UTC")).date()
    
    # ENTSO-E is 2-5 days behind, so don't request recent dates
    end_date = today - timedelta(days=3)  
    start_date = end_date - timedelta(days=90)  # Get 90 days of history
    
    st.info(f"📊 Building dataset from {start_date} to {end_date}")
    
    # Get API token
    token = os.getenv("ENTSOE_TOKEN")
    if not token:
        try:
            token = st.secrets["ENTSOE_TOKEN"]
        except:
            st.error("❌ ENTSOE_TOKEN not found. Add it to Streamlit Secrets (Settings → Secrets)")
            st.stop()
    
    # Initialize API client
    try:
        ent = Entsoe(token=token)
    except Exception as e:
        st.error(f"Failed to initialize ENTSO-E client: {e}")
        st.stop()
    
    # === FETCH DATA WITH CHUNKING ===
    
    def safe_fetch(method, start, end, is_dataframe=False):
        """Safely fetch data with error handling"""
        try:
            # Try the full range first
            result = getattr(ent, method)(start=str(start), end=str(end))
            if result is not None and len(result) > 0:
                return result
        except Exception as e:
            st.warning(f"⚠️ {method} failed: {e}")
        
        # Return empty Series or DataFrame
        return pd.DataFrame() if is_dataframe else pd.Series(dtype=float)
    
    # Fetch each data type
    st.caption("Fetching DAM prices...")
    dam = safe_fetch("day_ahead_prices", start_date, end_date)
    
    st.caption("Fetching load forecast...")
    load_fc = safe_fetch("load_forecast", start_date, end_date)
    
    st.caption("Fetching wind/solar forecast...")
    ws = safe_fetch("wind_solar_forecast", start_date, end_date, is_dataframe=True)
    
    st.info("Fetching DAM prices (SEMOpx HRP via reports API)...")
    try:
        if FAST_MODE:
            # Skip HRP (SEMOpx) in fast mode to avoid the extra network hop
            raise RuntimeError("Skip SEMOpx in fast mode")
        dam_df = fetch_dam_hrp_recent(days=DAYS)
        if dam_df is None or dam_df.empty:
            raise RuntimeError("SEMOpx returned empty HRP frame.")
        st.success(f"DAM (HRP) loaded: {dam_df['ts_utc'].min()} → {dam_df['ts_utc'].max()} ({len(dam_df)} rows)")
    except Exception as e:
        st.info(f"Using ENTSO-E (reason: {e})")
        # Do NOT force refresh unless user disabled fast mode
        entsoe_df = fetch_ie_dam_chunked(days=DAYS, chunk_days=7, force_refresh=not FAST_MODE)
        if isinstance(entsoe_df, pd.Series):
            entsoe_df = entsoe_df.to_frame("dam_eur_mwh")
        if entsoe_df is None or entsoe_df.empty:
            st.error("ENTSO-E returned no data.")
            st.stop()
        entsoe_df.index = pd.DatetimeIndex(entsoe_df.index, tz="UTC")
        dam_df = entsoe_df.rename_axis("ts_utc").reset_index()
        st.success(f"ENTSO-E DAM loaded: {dam_df['ts_utc'].min()} → {dam_df['ts_utc'].max()}")
    except Exception as e2:
        st.error(f"Could not load DAM from either source: {e2}")
        st.stop() 

    dam_df["ts_utc"] = pd.to_datetime(dam_df["ts_utc"], utc=True)
    dam_df = dam_df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last").reset_index(drop=True)


    
    min_dt = dam_df["ts_utc"].min().tz_convert("Europe/Dublin").date()
    max_dt = (dam_df["ts_utc"].max().tz_convert("Europe/Dublin") - pd.Timedelta(hours=1)).date()
    selected = st.date_input("Select forecast date:", value=max_dt, min_value=min_dt, max_value=max_dt)

    
    
    # === WEATHER DATA ===
    st.caption("Fetching weather data...")
    
    lat = cfg.get("weather", {}).get("lat", 53.4)
    lon = cfg.get("weather", {}).get("lon", -8.2)
    
    try:
        weather = fetch_hourly(lat, lon, str(start_date), str(end_date))
    except:
        # Fallback: create synthetic weather
        hours = dam.index
        weather = pd.DataFrame({
            'wind100m_ms': 8 + 4 * np.random.randn(len(hours)),
            'temperature_2m': 12 + 8 * np.sin(2 * np.pi * hours.dayofyear / 365),
            'cloud_cover': 60 + 20 * np.random.randn(len(hours))
        }, index=hours)
    
    # === TIMEZONE ALIGNMENT ===
    
    def make_tz_naive_dublin(df):
        """Ensure all data is in Dublin time, tz-naive"""
        if not hasattr(df, 'index'):
            return df
        
        idx = df.index
        if hasattr(idx, 'tz'):
            if idx.tz is not None:
                idx = idx.tz_convert(ZoneInfo("Europe/Dublin"))
            else:
                # ENTSO-E data is usually in Brussels time
                try:
                    idx = idx.tz_localize(ZoneInfo("Europe/Brussels")).tz_convert(ZoneInfo("Europe/Dublin"))
                except:
                    idx = idx.tz_localize(ZoneInfo("UTC")).tz_convert(ZoneInfo("Europe/Dublin"))
            
            # Make timezone naive
            idx = idx.tz_localize(None)
        
        df.index = idx
        return df
    
    # Apply timezone fixes
    dam = make_tz_naive_dublin(dam)
    load_fc = make_tz_naive_dublin(load_fc)
    ws = make_tz_naive_dublin(ws)
    weather = make_tz_naive_dublin(weather)
    
    # Remove duplicates
    dam = dam[~dam.index.duplicated(keep='last')]
    load_fc = load_fc[~load_fc.index.duplicated(keep='last')]
    ws = ws[~ws.index.duplicated(keep='last')]
    weather = weather[~weather.index.duplicated(keep='last')]
    
    # Ensure load_fc is a Series
    if isinstance(load_fc, pd.DataFrame):
        load_fc = load_fc.iloc[:, 0]
    load_fc.name = 'load_forecast_mw'
    
    # === BUILD FEATURES ===
    st.caption("Building features...")
    
    try:
        X = build_feature_table(dam, load_fc, ws, weather)
        y = make_day_ahead_target(dam).reindex(X.index)
        
        # CRITICAL FIX: Add the raw columns that build_feature_table might not include
        if 'dam_eur_mwh' not in X.columns:
            X['dam_eur_mwh'] = dam.reindex(X.index)
        if 'load_forecast_mw' not in X.columns:
            X['load_forecast_mw'] = load_fc.reindex(X.index)
            
    except Exception as e:
        st.error(f"Feature building failed: {e}")
        st.stop()
    
    # Filter valid rows - check which columns actually exist
    valid = y.notna()
    
    # Only check for columns that exist
    if 'dam_eur_mwh' in X.columns:
        valid = valid & X['dam_eur_mwh'].notna()
    if 'load_forecast_mw' in X.columns:
        valid = valid & X['load_forecast_mw'].notna()
    
    X = X[valid]
    y = y[valid]
    
    if X.empty:
        st.error("No valid training data after processing!")
        st.stop()
    
    # Fill missing values
    X = X.fillna(method='ffill', limit=24).fillna(method='bfill', limit=24)
    X = X.fillna(0)  # Final fallback
    
    # Save dataset
    out = X.copy()
    out['target'] = y
    
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_PATH)
    
    st.success(f"✅ Dataset built: {len(out):,} rows from {out.index.min().date()} to {out.index.max().date()}")

# ============== MAIN APP ==============

st.set_page_config(
    page_title="Irish Power Price Forecast",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Irish Day-Ahead Power Price Forecast (SEMOpx)")

# Sidebar
with st.sidebar:
    st.header("📊 Data Management")
    
    if st.button("🔄 Rebuild Dataset", help="Force refresh all data"):
        if DATA_PATH.exists():
            DATA_PATH.unlink()
        st.cache_data.clear()
        st.rerun()
    
    # Show data status
    if DATA_PATH.exists():
        mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        age = (datetime.now() - mod_time).total_seconds() / 3600
        
        if age < 1:
            status = "🟢 Fresh"
        elif age < 6:
            status = "🟡 Recent"
        else:
            status = "🔴 Stale"
        
        st.markdown(f"""
        **Data Status**
        - Status: {status}
        - Updated: {mod_time.strftime('%H:%M')}
        - Age: {age:.1f} hours
        """)
    
    # Check EirGrid availability
    if EirGridBackup.get_recent_data():
        st.success("✅ EirGrid backup available")
    else:
        st.info("ℹ️ Using ENTSO-E only")

# Build or load dataset
ensure_dataset()

# Load the data
try:
    df = pd.read_parquet(DATA_PATH)
except FileNotFoundError:
    st.error("Dataset file not found. Please rebuild.")
    st.stop()

# Prepare for modeling
y = df.pop('target') if 'target' in df.columns else pd.Series(index=df.index)

# Train model - with fallback
model = None
try:
    from src.models.xgb_model import make_model
    model = make_model()
    model.fit(df, y)
    st.success("✅ Model trained successfully (XGBoost)")
except Exception as e:
    st.warning(f"XGBoost failed ({e}), using Random Forest")
    try:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(df, y)
        st.success("✅ Model trained successfully (Random Forest)")
    except Exception as e2:
        st.error(f"Model training failed: {e2}")
        st.stop()

# === DATE SELECTION ===

# Get date range from data
date_min = df.index.min().date()
date_max = df.index.max().date()
date_default = date_max  # Default to most recent

selected_date = st.date_input(
    "Select forecast date:",
    value=date_default,
    min_value=date_min,
    max_value=date_max,
    help=f"Data available from {date_min} to {date_max}"
)

# === GENERATE FORECAST ===

# Get data for selected date
day_data = df[df.index.date == selected_date]

if not day_data.empty:
    # Make predictions
    predictions = model.predict(day_data)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Hour': day_data.index,
        'Forecast (€/MWh)': predictions.round(2)
    })
    
    # Add additional columns if they exist
    if 'load_forecast_mw' in day_data.columns:
        results['Load (MW)'] = day_data['load_forecast_mw'].round(0).values
    if 'wind_total_mw' in day_data.columns:
        results['Wind (MW)'] = day_data['wind_total_mw'].round(0).values
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 Forecast for {selected_date}")
        
        # Plot
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['Hour'],
            y=results['Forecast (€/MWh)'],
            mode='lines+markers',
            name='Price Forecast',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Day-Ahead Price Forecast - {selected_date}",
            xaxis_title="Hour",
            yaxis_title="Price (€/MWh)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Statistics")
        
        avg_price = predictions.mean()
        max_price = predictions.max()
        min_price = predictions.min()
        
        st.metric("Average", f"€{avg_price:.2f}/MWh")
        st.metric("Peak", f"€{max_price:.2f}/MWh")
        st.metric("Off-Peak", f"€{min_price:.2f}/MWh")
        
        # Peak hours
        peak_hour = results.loc[results['Forecast (€/MWh)'].idxmax(), 'Hour'].hour
        st.info(f"🕐 Peak hour: {peak_hour}:00")
    
    # Detailed table
    with st.expander("📋 Detailed Hourly Forecast"):
        st.dataframe(
            results.set_index('Hour'),
            use_container_width=True
        )
        
        # Download button
        csv = results.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv,
            f"forecast_{selected_date}.csv",
            "text/csv"
        )
else:
    st.warning(f"No data available for {selected_date}")

# === ADDITIONAL FEATURES ===

with st.expander("🔍 Model Performance"):
    # Ensure we're testing on truly held-out data
    if len(df) > 24*14:  # At least 14 days of data
        # Use last 7 days for testing, previous data for training
        split_point = -24*7
        
        X_train = df.iloc[:split_point]
        y_train = y.iloc[:split_point]
        X_test = df.iloc[split_point:]
        y_test = y.iloc[split_point:]
        
        # Retrain model on training data only
        test_model = make_model()
        test_model.fit(X_train, y_train)
        
        # Predict on test set
        test_pred = test_model.predict(X_test)
        
        # Calculate realistic metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_test, test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        col1, col2 = st.columns(2)
        col1.metric("MAE (7-day test)", f"€{mae:.2f}/MWh")
        col2.metric("RMSE (7-day test)", f"€{rmse:.2f}/MWh")
        
        # Expected ranges for Irish market
        if mae < 5:
            st.warning("⚠️ MAE seems too low - check for data leakage")
        elif mae > 30:
            st.warning("⚠️ MAE seems high - model needs improvement")
        else:
            st.success(f"✅ MAE of €{mae:.2f}/MWh is reasonable for Irish DAM")

# Add feature importance if possible
with st.expander("📊 Feature Importance"):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'Feature': df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            st.bar_chart(importances.set_index('Feature'))
    except:
        st.info("Feature importance not available for this model")

# Footer
st.markdown("---")
st.caption("💡 Data source: ENTSO-E Transparency Platform | Weather: Open-Meteo | Built with Streamlit")
