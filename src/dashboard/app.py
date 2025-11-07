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

# ---- Third-party
import pandas as pd
import numpy as np
import streamlit as st
import yaml
import requests

# ---- Project imports (use 'data.*' / 'features.*' with our path bootstrap)
from data.semopx_api import fetch_dam_hrp_recent
from data.entsoe_api import fetch_ie_dam_recent, fetch_ie_dam_chunked, Entsoe
from entsoe.exceptions import NoMatchingDataError
from data.weather import fetch_hourly
from features.build_features import build_feature_table
from features.targets import make_day_ahead_target

# -------------------- Page / Sidebar --------------------
st.set_page_config(
    page_title="Irish Power Price Forecast",
    page_icon="⚡",
    layout="wide"
)

# Sidebar toggles
FAST_MODE = st.sidebar.checkbox("⚡ Fast mode (use cache, skip SEMOpx if slow)", value=True)
DAYS = st.sidebar.slider("History window (days)", 7, 60, 21)

# -------------------- Caching wrappers --------------------
@st.cache_data(ttl=60*30, show_spinner=False)
def build_dam_cached(fast_mode: bool, days: int) -> pd.DataFrame:
    """
    Return a tidy DAM dataframe with columns ['ts_utc','dam_eur_mwh'].
    Prefer SEMOpx HRP unless fast_mode is True or HRP fails; fallback to ENTSO-E chunked (delay_days=3).
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
        # Fallback to ENTSO-E (do NOT force refresh in fast mode)
        try:
            entsoe_df = fetch_ie_dam_chunked(days=days, chunk_days=7, force_refresh=not fast_mode, delay_days=3)
        except NoMatchingDataError as e2:
            # As an extra fallback, try recent (which also uses delay_days internally)
            entsoe_df = fetch_ie_dam_recent(days=days, force_refresh=not fast_mode, delay_days=3)
        if isinstance(entsoe_df, pd.Series):
            entsoe_df = entsoe_df.to_frame("dam_eur_mwh")
        if entsoe_df is None or entsoe_df.empty:
            raise RuntimeError("ENTSO-E returned no data.")
        entsoe_df.index = pd.DatetimeIndex(entsoe_df.index, tz="UTC")
        return entsoe_df.rename_axis("ts_utc").reset_index()

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
DATA_PATH = Path("data/processed/train.parquet")

@st.cache_data(show_spinner=True, ttl=3600)
def ensure_dataset():
    """Build the training dataset with robust error handling."""
    # Quick freshness check
    if DATA_PATH.exists():
        mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(hours=6):
            return  # Use cached file

    # Load config (lat/lon for weather)
    try:
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    lat = cfg.get("weather", {}).get("lat", 53.4)
    lon = cfg.get("weather", {}).get("lon", -8.2)

    # ---- DAM prices (cached + robust fallback) ----
    st.info("Fetching DAM prices…")
    dam_df = build_dam_cached(FAST_MODE, DAYS)  # ['ts_utc','dam_eur_mwh']
    dam_df["ts_utc"] = pd.to_datetime(dam_df["ts_utc"], utc=True)
    dam_df = dam_df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last").reset_index(drop=True)

    # Date window for fundamentals/forecast (use the DAM range in Dublin time)
    min_dt_local = dam_df["ts_utc"].min().tz_convert("Europe/Dublin")
    max_dt_local = dam_df["ts_utc"].max().tz_convert("Europe/Dublin")
    start_local = (min_dt_local).normalize()
    end_local   = (max_dt_local).normalize()

    st.caption(f"Window for fundamentals: {start_local.date()} → {end_local.date()}")

    # ---- Fundamentals (cached) ----
    funds = build_fundamentals_cached(start_local, end_local, lat, lon)
    load_fc = funds["load_fc"]
    ws_fc   = funds["ws_fc"]
    weather = funds["weather"]

    # ---- Convert DAM to Series aligned with rest (Dublin tz-naive) ----
    dam_series = dam_df.set_index("ts_utc")["dam_eur_mwh"]
    # Convert to Europe/Dublin, then make tz-naive for feature builder
    dam_series = dam_series.tz_convert(ZoneInfo("Europe/Dublin")).tz_localize(None)

    def _to_dublin_naive(x):
        if isinstance(x, pd.Series):
            idx = x.index
        else:
            idx = x.index if hasattr(x, "index") else None
        if idx is None:
            return x
        # localize/convert
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(ZoneInfo("Europe/Dublin")).tz_localize(None)
        else:
            # assume UTC if naive
            idx = idx.tz_localize(ZoneInfo("UTC")).tz_convert(ZoneInfo("Europe/Dublin")).tz_localize(None)
        x.index = idx
        return x

    load_fc = _to_dublin_naive(load_fc)
    ws_fc   = _to_dublin_naive(ws_fc)
    weather = _to_dublin_naive(weather)

    # Remove duplicates (safety)
    dam_series = dam_series[~dam_series.index.duplicated(keep="last")]
    if isinstance(load_fc, pd.Series):
        load_fc = load_fc[~load_fc.index.duplicated(keep="last")]
    if isinstance(ws_fc, pd.DataFrame):
        ws_fc = ws_fc[~ws_fc.index.duplicated(keep="last")]
    if isinstance(weather, pd.DataFrame):
        weather = weather[~weather.index.duplicated(keep="last")]

    # ---- Build features/target (cached) ----
    st.caption("Building features…")
    try:
        X, y = build_features_cached(dam_series, load_fc, ws_fc, weather)
        # Safety: ensure key raw columns exist if your feature builder omitted them
        if "dam_eur_mwh" not in X.columns:
            X["dam_eur_mwh"] = dam_series.reindex(X.index)
        if "load_forecast_mw" not in X.columns and isinstance(load_fc, pd.Series):
            X["load_forecast_mw"] = load_fc.reindex(X.index)
    except Exception as e:
        st.error(f"Feature building failed: {e}")
        st.stop()

    # ---- Filter valid rows ----
    valid = y.notna()
    if "dam_eur_mwh" in X.columns:
        valid &= X["dam_eur_mwh"].notna()
    if "load_forecast_mw" in X.columns:
        valid &= X["load_forecast_mw"].notna()

    X = X[valid]
    y = y[valid]

    if X.empty:
        st.error("No valid training rows after processing.")
        st.stop()

    # ---- Impute remaining gaps ----
    X = X.fillna(method="ffill", limit=24).fillna(method="bfill", limit=24).fillna(0)

    # ---- Persist dataset ----
    out = X.copy()
    out["target"] = y
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_PATH)

    st.success(f"✅ Dataset built: {len(out):,} rows "
               f"from {out.index.min().date()} to {out.index.max().date()}")

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
