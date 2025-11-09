# src/dashboard/app.py
# --- Path bootstrap: make `src/` importable when running as a script ---
import os, sys
from pathlib import Path

_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR  = os.path.abspath(os.path.join(_THIS_DIR, ".."))   # -> .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---- Standard libs
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import time  # <-- add this

# ---- Third-party
import pandas as pd
import numpy as np
import streamlit as st
import yaml
import requests

# ---- Project imports (use 'data.*' / 'features.*' with our path bootstrap)
# from data.semopx_api import fetch_dam_hrp_recent
from data.eirgrid_prices import fetch_dam_recent  
# from data.entsoe_api import fetch_ie_dam_recent, fetch_ie_dam_chunked, Entsoe
from entsoe.exceptions import NoMatchingDataError
from data.weather import fetch_hourly
from features.build_features import build_feature_table
from features.targets import make_day_ahead_target
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# -------------------- Page / Sidebar --------------------
st.set_page_config(
    page_title="Irish Power Price Forecast",
    page_icon="⚡",
    layout="wide"
)

# Sidebar toggles
# FAST_MODE = st.sidebar.checkbox("⚡ Fast mode (use cache, skip SEMOpx if slow)", value=True)
DAYS = 21
TIME_BUDGET = 63
st.sidebar.info("Using synthetic demo data (no live market prices).")


# -------------------- Caching wrappers --------------------
@st.cache_data(ttl=60*30, show_spinner=False)
def build_dam_cached(fast_mode: bool, days: int) -> pd.DataFrame:
    """
    Fetch Irish DAM prices from EirGrid (Smart Grid Dashboard).
    ENTSO-E and SEMOpx are NOT used here.
    """
    df = fetch_dam_recent(days=days, force_refresh=not fast_mode)

    # df is indexed by UTC; turn it into a clean ['ts_utc','dam_eur_mwh'] table
    if isinstance(df, pd.Series):
        df = df.to_frame("dam_eur_mwh")

    df = df.copy()
    if "ts_utc" not in df.columns:
        df["ts_utc"] = df.index

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")

    return df.rename_axis("ts_utc").reset_index()


@st.cache_data(ttl=60*30, show_spinner=False)
def build_fundamentals_cached(start_local: pd.Timestamp, end_local: pd.Timestamp, lat: float, lon: float):
    """
    Pull load forecast, wind/solar forecast, and weather; return as dict.
    Cached so we don’t refetch on every UI change.
    """
    e = Entsoe()  # uses ENTSOE_TOKEN
    # ENTSO-E fundamentals
    try:
        load_fc = e.load_forecast(start_local, end_local)           # Series (tz-aware)
    except Exception:
        load_fc = pd.Series(dtype=float, name="load_forecast_mw")
    try:
        ws_fc   = e.wind_solar_forecast(start_local, end_local)     # DataFrame
    except Exception:
        ws_fc = pd.DataFrame()

    # Weather (Open-Meteo in your project)
    try:
        weather = fetch_hourly(lat, lon, str(start_local.date()), str(end_local.date()))
    except Exception:
        weather = pd.DataFrame()

    return {"load_fc": load_fc, "ws_fc": ws_fc, "weather": weather}

@st.cache_data(ttl=60*30, show_spinner=False)
def build_features_cached(dam_series: pd.Series, load_fc: pd.Series, ws_fc: pd.DataFrame, weather: pd.DataFrame):
    """
    Build X and y once and cache the result.
    """
    y = make_day_ahead_target(dam_series)             # Series (target aligned to delivery)
    X = build_feature_table(dam_series, load_fc, ws_fc, weather)  # DataFrame
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

# -------------------- EirGrid Backup indicator (UI only) --------------------
class EirGridBackup:
    """Light indicator if Smart Grid Dashboard responds (not used for build)."""
    @staticmethod
    def get_recent_data() -> bool:
        try:
            base_url = "https://www.smartgriddashboard.com/DashboardService.svc/data"
            date_from = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
            date_to = datetime.now().strftime("%d-%b-%Y")
            params = {"area": "windactual", "region": "ALL", "datefrom": date_from, "dateto": date_to}
            resp = requests.get(base_url, params=params, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

# -------------------- Dataset build --------------------
# -------------------- Dataset build --------------------
DATA_PATH = Path("data/processed/train.parquet")

def ensure_dataset():
    """Build the training dataset with robust progress + time budget."""
    t0 = time.perf_counter()

    def over_budget() -> bool:
        return (time.perf_counter() - t0) > TIME_BUDGET

    with st.status("Building dataset…", expanded=True) as status:
        # Quick freshness: re-use file if <6h old
        if DATA_PATH.exists():
            mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(hours=6):
                st.write("✅ Using cached dataset on disk (fresh < 6h).")
                status.update(label="Done", state="complete")
                return

        # Load config (lat/lon for weather)
        try:
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        lat = cfg.get("weather", {}).get("lat", 53.4)
        lon = cfg.get("weather", {}).get("lon", -8.2)

        # -------- DAM prices (cached wrappers) --------
        st.write("🔹 Fetching DAM prices…")
        dam_df = None

        def _get_dam():
            # call the cached wrapper; it may still do network I/O
            return build_dam_cached(False, DAYS)  # ['ts_utc','dam_eur_mwh']
            
        try:
            # use 80% of the total budget for DAM; the rest is for fundamentals/features
            dam_timeout = max(5, int(TIME_BUDGET * 0.8))
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_get_dam)
                dam_df = fut.result(timeout=dam_timeout)

        except FuturesTimeout:
            st.write("⏳ DAM fetch exceeded time budget.")
            # If we have a previous dataset on disk, use it and stop the build early.
            if DATA_PATH.exists():
                st.write("↩️ Using last saved dataset instead.")
                status.update(label="Done (fallback to cached file)", state="complete")
                return
            # No cached file: fall back to a tiny, synthetic dataset so UI stays responsive.
            # (This keeps the app usable; you can rebuild later with a larger budget.)
            st.write("🛟 No cached file. Creating minimal synthetic dataset.")
            end_local = pd.Timestamp.now(tz="Europe/Dublin").floor("H")
            idx = pd.date_range(end=end_local, periods=DAYS * 24, freq="H")
            base = 80 + 10 * np.sin(2 * np.pi * (idx.hour / 24.0))  # simple diurnal curve
            dam_series = pd.Series(base, index=idx, name="dam_eur_mwh")   

            # Minimal features + target
            X_min = pd.DataFrame({
                "hour": idx.hour,
                "dow": idx.dayofweek,
                "month": idx.month,
                "is_peak": ((idx.hour >= 17) & (idx.hour <= 19)).astype(int),
                "dam_eur_mwh": dam_series.values
            }, index=idx)
            y_min = make_day_ahead_target(dam_series).reindex(X_min.index)

            # 🔧 drop rows with invalid target (last 24h etc.)
            mask = y_min.notna() & np.isfinite(y_min)
            X_min = X_min[mask]
            y_min = y_min[mask]

            out = X_min.copy()
            out["target"] = y_min
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(DATA_PATH)
            status.update(label="Done (minimal synthetic)", state="complete")
            return   

        except Exception as e:
            st.write(f"⚠️ DAM fetch failed: {e}")
            if DATA_PATH.exists():
                st.write("↩️ Falling back to last saved dataset.")
                status.update(label="Done (fallback to cached file)", state="complete")
                return

            # 🛟 No cached file -> synthetic minimal dataset so UI still works
            st.write("🛟 No DAM data and no cache. Creating minimal synthetic dataset.")
            end_local = pd.Timestamp.now(tz="Europe/Dublin").floor("H")
            idx = pd.date_range(end=end_local, periods=DAYS * 24, freq="H")
            base = 80 + 10 * np.sin(2 * np.pi * (idx.hour / 24.0))
            dam_series = pd.Series(base, index=idx, name="dam_eur_mwh")

            X_min = pd.DataFrame({
                "hour": idx.hour,
                "dow": idx.dayofweek,
                "month": idx.month,
                "is_peak": ((idx.hour >= 17) & (idx.hour <= 19)).astype(int),
                "dam_eur_mwh": dam_series.values
            }, index=idx)
            y_min = make_day_ahead_target(dam_series).reindex(X_min.index)

            # 🔧 drop rows with invalid target
            mask = y_min.notna() & np.isfinite(y_min)
            X_min = X_min[mask]
            y_min = y_min[mask]

            out = X_min.copy()
            out["target"] = y_min
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(DATA_PATH)
            status.update(label="Done (minimal synthetic)", state="complete")
            return

        # -------- Normal path continues here --------
        dam_df["ts_utc"] = pd.to_datetime(dam_df["ts_utc"], utc=True)
        dam_df = dam_df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last").reset_index(drop=True)

        # Prepare DAM series in Dublin tz-naive (what your feature builder expects)
        dam_series = dam_df.set_index("ts_utc")["dam_eur_mwh"].tz_convert(ZoneInfo("Europe/Dublin")).tz_localize(None)

        # If we’re already over budget, write a minimal dataset and exit
        if over_budget():
            st.write("⏳ Time budget reached after DAM — writing minimal dataset.")
            idx = dam_series.index
            X_min = pd.DataFrame({
                "hour": idx.hour,
                "dow": idx.dayofweek,
                "month": idx.month,
                "is_peak": ((idx.hour >= 17) & (idx.hour <= 19)).astype(int),
                "dam_eur_mwh": dam_series.values
            }, index=idx)
            y_min = make_day_ahead_target(dam_series).reindex(X_min.index)

            # 🔧 drop rows with invalid target
            mask = y_min.notna() & np.isfinite(y_min)
            X_min = X_min[mask]
            y_min = y_min[mask]

            out = X_min.copy()
            out["target"] = y_min
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(DATA_PATH)
            status.update(label="Done (minimal features)", state="complete")
            return

        # -------- Fundamentals window from DAM range --------
        min_dt_local = dam_df["ts_utc"].min().tz_convert("Europe/Dublin")
        max_dt_local = dam_df["ts_utc"].max().tz_convert("Europe/Dublin")
        start_local = min_dt_local.normalize()
        end_local   = max_dt_local.normalize()

        st.write(f"🔹 Fundamentals window: {start_local.date()} → {end_local.date()}")

        # -------- Fundamentals (cached) --------
        st.write("🔹 Fetching load/wind/solar/weather…")
        funds = build_fundamentals_cached(start_local, end_local, lat, lon)
        load_fc = funds["load_fc"]
        ws_fc   = funds["ws_fc"]
        weather = funds["weather"]

        # Convert to Dublin tz-naive to align with your feature builder
        def _to_dublin_naive(x):
            if not hasattr(x, "index"):
                return x
            idx = x.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert(ZoneInfo("Europe/Dublin")).tz_localize(None)
            else:
                idx = idx.tz_localize(ZoneInfo("UTC")).tz_convert(ZoneInfo("Europe/Dublin")).tz_localize(None)
            x.index = idx
            return x

        load_fc = _to_dublin_naive(load_fc)
        ws_fc   = _to_dublin_naive(ws_fc)
        weather = _to_dublin_naive(weather)

        # Deduplicate indices
        dam_series = dam_series[~dam_series.index.duplicated(keep="last")]
        if hasattr(load_fc, "index"):
            load_fc = load_fc[~load_fc.index.duplicated(keep="last")]
        if hasattr(ws_fc, "index"):
            ws_fc = ws_fc[~ws_fc.index.duplicated(keep="last")]
        if hasattr(weather, "index"):
            weather = weather[~weather.index.duplicated(keep="last")]

        # If over budget here, fall back to minimal features
        if over_budget():
            st.write("⏳ Time budget reached during fundamentals — writing minimal dataset.")
            idx = dam_series.index
            X_min = pd.DataFrame({
                "hour": idx.hour,
                "dow": idx.dayofweek,
                "month": idx.month,
                "is_peak": ((idx.hour >= 17) & (idx.hour <= 19)).astype(int),
                "dam_eur_mwh": dam_series.values
            }, index=idx)
            y_min = make_day_ahead_target(dam_series).reindex(X_min.index)

            # 🔧 drop rows with invalid target
            mask = y_min.notna() & np.isfinite(y_min)
            X_min = X_min[mask]
            y_min = y_min[mask]

            out = X_min.copy()
            out["target"] = y_min
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(DATA_PATH)
            status.update(label="Done (minimal features)", state="complete")
            return

        # -------- Build features / target --------
        st.write("🔹 Building features…")
        try:
            X, y = build_features_cached(dam_series, load_fc, ws_fc, weather)
            if "dam_eur_mwh" not in X.columns:
                X["dam_eur_mwh"] = dam_series.reindex(X.index)
            if "load_forecast_mw" not in X.columns and hasattr(load_fc, "reindex"):
                X["load_forecast_mw"] = load_fc.reindex(X.index)
        except Exception as e:
            st.write(f"⚠️ Feature build failed: {e} — falling back to minimal features.")
            idx = dam_series.index
            X = pd.DataFrame({
                "hour": idx.hour,
                "dow": idx.dayofweek,
                "month": idx.month,
                "is_peak": ((idx.hour >= 17) & (idx.hour <= 19)).astype(int),
                "dam_eur_mwh": dam_series.values
            }, index=idx)
            y = make_day_ahead_target(dam_series).reindex(X.index)

        # Filter valid rows
        valid = y.notna()
        if "dam_eur_mwh" in X.columns:
            valid &= X["dam_eur_mwh"].notna()
        
        X = X[valid]
        y = y[valid]
        if X.empty:
            st.error("No valid rows after processing.")
            st.stop()

        # Impute small gaps
        X = X.fillna(method="ffill", limit=24).fillna(method="bfill", limit=24).fillna(0)

        # Save dataset
        out = X.copy()
        out["target"] = y
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(DATA_PATH)

        status.update(label=f"Done — {len(out):,} rows saved", state="complete")




# -------------------- Sidebar: Data management --------------------
with st.sidebar:
    st.header("📊 Data Management")
    if st.button("🔄 Rebuild Dataset", help="Force refresh all data"):
        if DATA_PATH.exists():
            DATA_PATH.unlink()
        st.cache_data.clear()
        st.rerun()

    # Status
    if DATA_PATH.exists():
        mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        age = (datetime.now() - mod_time).total_seconds() / 3600
        if age < 1:
            status = "🟢 Fresh"
        elif age < 6:
            status = "🟡 Recent"
        else:
            status = "🔴 Stale"
        st.markdown(f"**Data Status**\n\n- Status: {status}\n- Updated: {mod_time.strftime('%H:%M')}\n- Age: {age:.1f} h")

    if EirGridBackup.get_recent_data():
        st.success("✅ EirGrid backup available")
    else:
        st.info("ℹ️ Using ENTSO-E only")

# -------------------- Build/load dataset --------------------
ensure_dataset()

try:
    df = pd.read_parquet(DATA_PATH)
except FileNotFoundError:
    st.error("Dataset file not found. Please rebuild.")
    st.stop()

# -------------------- Train model --------------------
y = df.pop("target") if "target" in df.columns else pd.Series(index=df.index)

model = None
try:
    # Import with repo root on sys.path (path bootstrap at the top adds it)
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from src.models.xgb_model import make_model  # project helper if present
    model = make_model()
    model.fit(df, y)
    st.success("✅ Model trained successfully (XGBoost helper)")
except Exception as e:
    st.warning(f"XGBoost helper failed ({e}); using plain XGBoost")
    try:
        model = fit_model_cached(df, y)
        st.success("✅ Model trained successfully (XGBoost)")
    except Exception as e2:
        st.warning(f"XGBoost failed ({e2}); using Random Forest")
        try:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
            model.fit(df, y)
            st.success("✅ Model trained successfully (Random Forest)")
        except Exception as e3:
            st.error(f"Model training failed: {e3}")
            st.stop()

# -------------------- UI: date selection --------------------
st.title("⚡ Irish Day-Ahead Power Price Forecast (SEMOpx/ENTSO-E)")

date_min = df.index.min().date()
date_max = df.index.max().date()
selected_date = st.date_input(
    "Select forecast date:",
    value=date_max,
    min_value=date_min,
    max_value=date_max,
    help=f"Data available from {date_min} to {date_max}"
)

# -------------------- Forecast for selected date --------------------
day_data = df[df.index.date == selected_date]

if not day_data.empty:
    preds = model.predict(day_data)

    results = pd.DataFrame({
        "Hour": day_data.index,
        "Forecast (€/MWh)": preds.round(2)
    })

    if "load_forecast_mw" in day_data.columns:
        results["Load (MW)"] = day_data["load_forecast_mw"].round(0).values
    if "wind_total_mw" in day_data.columns:
        results["Wind (MW)"] = day_data["wind_total_mw"].round(0).values

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📈 Forecast for {selected_date}")
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results["Hour"],
            y=results["Forecast (€/MWh)"],
            mode="lines+markers",
            name="Price Forecast",
            line=dict(width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f"Day-Ahead Price Forecast - {selected_date}",
            xaxis_title="Hour",
            yaxis_title="Price (€/MWh)",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Statistics")
        avg_price = float(np.mean(preds))
        max_price = float(np.max(preds))
        min_price = float(np.min(preds))
        st.metric("Average", f"€{avg_price:.2f}/MWh")
        st.metric("Peak", f"€{max_price:.2f}/MWh")
        st.metric("Off-Peak", f"€{min_price:.2f}/MWh")
        peak_hour = results.loc[results["Forecast (€/MWh)"].idxmax(), "Hour"].hour
        st.info(f"🕐 Peak hour: {peak_hour}:00")

    with st.expander("📋 Detailed Hourly Forecast"):
        st.dataframe(results.set_index("Hour"), use_container_width=True)
        csv = results.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, f"forecast_{selected_date}.csv", "text/csv")
else:
    st.warning(f"No data available for {selected_date}")

# -------------------- Performance section --------------------
with st.expander("🔍 Model Performance"):
    if len(df) > 24 * 14:  # at least 14 days of data
        split_point = -24 * 7
        X_train = df.iloc[:split_point]
        y_train = y.iloc[:split_point]
        X_test  = df.iloc[split_point:]
        y_test  = y.iloc[split_point:]

        # Try helper; fall back to xgb
        try:
            from src.models.xgb_model import make_model
            test_model = make_model()
        except Exception:
            test_model = XGBRegressor(
                n_estimators=600, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                tree_method="hist", objective="reg:squarederror"
            )

        test_model.fit(X_train, y_train)
        test_pred = test_model.predict(X_test)

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_test, test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))

        c1, c2 = st.columns(2)
        c1.metric("MAE (7-day test)", f"€{mae:.2f}/MWh")
        c2.metric("RMSE (7-day test)", f"€{rmse:.2f}/MWh")

        if mae < 5:
            st.warning("⚠️ MAE seems too low — check for data leakage")
        elif mae > 30:
            st.warning("⚠️ MAE seems high — model needs improvement")
        else:
            st.success(f"✅ MAE of €{mae:.2f}/MWh looks reasonable for Irish DAM")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("💡 Data: SEMOpx (HRP) / ENTSO-E | Weather: Open-Meteo | Built with Streamlit")
