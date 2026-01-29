# src/dashboard/app.py
# FIXED VERSION - Syntax errors corrected, performance optimized
# --- Path bootstrap: make `src/` importable when running as a script ---
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from features.build_features import build_feature_table
from features.targets import make_day_ahead_target
from models.xgb_model import make_model

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Irish Power Price Forecast",
    page_icon="⚡",
    layout="wide"
)

RAW_PQ = Path("data/raw/semopx_dam_60min_hrp.parquet")
RAW_CSV = Path("data/raw/semopx_dam_60min_hrp.csv")
DATA_PATH = Path("data/processed/train.parquet")
IE_TZ = "Europe/Dublin"


# -------------------- Caching wrappers --------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def load_hrp60_local(days: int) -> pd.DataFrame:
    """Load HRP60 data from local parquet or CSV file."""
    if RAW_PQ.exists():
        df = pd.read_parquet(RAW_PQ)
    elif RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
    else:
        return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])

    if "dam_60min_hrp_eur_mwh" in df.columns and "dam_eur_mwh" not in df.columns:
        df = df.rename(columns={"dam_60min_hrp_eur_mwh": "dam_eur_mwh"})

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["dam_eur_mwh"] = pd.to_numeric(df["dam_eur_mwh"], errors="coerce")
    df = df.dropna(subset=["ts_utc", "dam_eur_mwh"]).sort_values("ts_utc")
    df = df.drop_duplicates("ts_utc", keep="last").reset_index(drop=True)

    if not df.empty:
        cutoff = df["ts_utc"].max() - pd.Timedelta(days=days)
        df = df[df["ts_utc"] >= cutoff].copy()

    return df[["ts_utc", "dam_eur_mwh"]]


@st.cache_resource(show_spinner="Training model...")
def train_model_cached(_X_hash: str, X: pd.DataFrame, y: pd.Series):
    """Train model with caching based on data hash."""
    m = make_model()
    m.fit(X, y)
    return m


def get_data_hash(df: pd.DataFrame) -> str:
    """Generate a hash for cache invalidation."""
    return f"{len(df)}_{df.index.max()}_{df.index.min()}"


# -------------------- Dataset build --------------------
def ensure_dataset(days: int):
    """Build or refresh the dataset if needed."""
    if DATA_PATH.exists():
        mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(minutes=10):
            return

    dam_df = load_hrp60_local(days)
    if dam_df.empty:
        # No HRP60 file -> synthetic fallback so UI still loads
        end_local = pd.Timestamp.now(tz=IE_TZ).floor("H")
        idx = pd.date_range(end=end_local, periods=days * 24, freq="H")
        base = 80 + 10 * np.sin(2 * np.pi * (idx.hour / 24.0))
        dam = pd.Series(base, index=idx, name="dam_eur_mwh")
    else:
        dam_df["ts_utc"] = pd.to_datetime(dam_df["ts_utc"], utc=True)
        dam = (
            dam_df.set_index("ts_utc")["dam_eur_mwh"]
            .tz_convert(IE_TZ)
            .tz_localize(None)
        )
        dam = dam[~dam.index.duplicated(keep="last")].sort_index()

    # No fundamentals in SEMOpx-only mode
    load_fc = pd.Series(index=dam.index, dtype=float, name="load_forecast_mw")
    ws_fc = pd.DataFrame(index=dam.index)
    weather = pd.DataFrame(index=dam.index)

    X = build_feature_table(dam, load_fc, ws_fc, weather)
    y = make_day_ahead_target(dam).reindex(X.index)

    out = X.copy()
    out["target"] = y
    out = out.dropna(subset=["target"]).ffill().bfill().fillna(0)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_PATH)


# -------------------- Sidebar: Data management --------------------
with st.sidebar:
    st.header("📊 Data")
    st.caption("This app reads local SEMOpx HRP60 CSV/Parquet only.")

    up = st.file_uploader("Upload HRP60 (CSV or Parquet)", type=["csv", "parquet"])
    if up is not None:
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        if up.name.endswith(".parquet"):
            RAW_PQ.write_bytes(up.getvalue())
            st.success("Saved parquet to data/raw/semopx_dam_60min_hrp.parquet")
        else:
            RAW_CSV.write_bytes(up.getvalue())
            st.success("Saved csv to data/raw/semopx_dam_60min_hrp.csv")
        st.cache_data.clear()
        st.rerun()

    if st.button("🔄 Rebuild Dataset"):
        if DATA_PATH.exists():
            DATA_PATH.unlink()
        st.cache_data.clear()
        st.rerun()

    # Dataset status - FIXED SYNTAX (was using escaped quotes)
    if DATA_PATH.exists():
        mod_time = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        age_h = (datetime.now() - mod_time).total_seconds() / 3600
        st.markdown(f"**Dataset**: updated {mod_time.strftime('%Y-%m-%d %H:%M')} ({age_h:.1f}h ago)")
    else:
        st.warning("train.parquet missing. App will build it from HRP60 file (if present).")

    DAYS = st.slider("History window (days)", 7, 365, 60)


# -------------------- Build/load dataset --------------------
ensure_dataset(DAYS)

try:
    df = pd.read_parquet(DATA_PATH)
except FileNotFoundError:
    st.error("Dataset file not found. Please rebuild.")
    st.stop()

# Sidebar: latest published DAM (from dataset)
with st.sidebar:
    if "dam_eur_mwh" in df.columns and len(df) > 0:
        last_ts = pd.Timestamp(df.index.max())
        last_val = float(df.loc[df.index.max(), "dam_eur_mwh"]) if df.index.max() in df.index else float("nan")
        st.markdown("### ⚡ Latest published DAM")
        st.write(f"**Time (delivery)**: {last_ts}")
        st.write(f"**Price**: €{last_val:.2f}/MWh")

# -------------------- Prepare data --------------------
y = df.pop("target") if "target" in df.columns else pd.Series(index=df.index)

# -------------------- UI: date selection --------------------
st.title("⚡ Irish Day-Ahead Power Price Forecast (SEMOpx HRP60)")

date_min = df.index.min().date()
date_max = df.index.max().date()
selected_date = st.date_input(
    "Select forecast date:",
    value=date_max,
    min_value=date_min,
    max_value=date_max,
    help=f"Data available from {date_min} to {date_max}"
)

# -------------------- Train model (no leakage) --------------------
train_mask = df.index.date < selected_date
test_mask = df.index.date == selected_date

X_train, y_train = df.loc[train_mask], y.loc[train_mask]
X_test = df.loc[test_mask]

if X_train.empty:
    st.error("Not enough historical data to train before the selected date.")
    st.stop()

# Train with spinner feedback
with st.spinner("Running train_model_cached(...)"):
    try:
        data_hash = get_data_hash(X_train)
        model = train_model_cached(data_hash, X_train, y_train)
        st.success("✅ Model trained successfully (cached)")
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

# -------------------- Forecast for selected date --------------------
day_data = X_test

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
        X_train_perf = df.iloc[:split_point]
        y_train_perf = y.iloc[:split_point]
        X_test_perf = df.iloc[split_point:]
        y_test_perf = y.iloc[split_point:]

        # Use optimized model
        try:
            test_model = make_model()
        except Exception:
            test_model = XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                tree_method="hist", objective="reg:squarederror"
            )

        test_model.fit(X_train_perf, y_train_perf)
        test_pred = test_model.predict(X_test_perf)

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_test_perf, test_pred)
        rmse = np.sqrt(mean_squared_error(y_test_perf, test_pred))

        c1, c2 = st.columns(2)
        c1.metric("MAE (7-day test)", f"€{mae:.2f}/MWh")
        c2.metric("RMSE (7-day test)", f"€{rmse:.2f}/MWh")

        if mae < 5:
            st.warning("⚠️ MAE seems too low — check for data leakage")
        elif mae > 30:
            st.warning("⚠️ MAE seems high — model needs improvement")
        else:
            st.success(f"✅ MAE of €{mae:.2f}/MWh looks reasonable for Irish DAM")

        # Actual vs Predicted chart
        perf_df = pd.DataFrame({
            "actual": y_test_perf,
            "predicted": test_pred,
        })
        perf_df = perf_df.sort_index()

        import plotly.graph_objects as go
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=perf_df.index,
            y=perf_df["actual"],
            mode="lines+markers",
            name="Actual"
        ))
        fig_perf.add_trace(go.Scatter(
            x=perf_df.index,
            y=perf_df["predicted"],
            mode="lines+markers",
            name="Predicted"
        ))
        fig_perf.update_layout(
            title="Actual vs Predicted (last 7 days)",
            xaxis_title="Delivery hour",
            yaxis_title="Price (€/MWh)",
            hovermode="x unified",
            height=350,
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        # Detailed errors table
        perf_df["abs_error"] = (perf_df["predicted"] - perf_df["actual"]).abs()
        with st.expander("📋 Last 7 days – detailed errors", expanded=False):
            st.dataframe(
                perf_df[["actual", "predicted", "abs_error"]]
                .rename(columns={
                    "actual": "Actual (€/MWh)",
                    "predicted": "Predicted (€/MWh)",
                    "abs_error": "|Error| (€/MWh)",
                }),
                use_container_width=True,
            )
    else:
        st.info("Need at least 14 days of data to compute a 7-day backtest.")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("💡 Data: SEMOpx (HRP60) | Built with Streamlit")
