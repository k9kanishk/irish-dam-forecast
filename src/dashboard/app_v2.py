# src/dashboard/app_v2.py
"""
Simplified Streamlit dashboard for Irish DAM price forecasting.
Works with the automated daily_update.py pipeline.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# === Paths ===
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FORECASTS_DIR = DATA_DIR / "forecasts"
RAW_DIR = DATA_DIR / "raw"

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
MODEL_FILE = MODELS_DIR / "latest_model.joblib"
FORECAST_FILE = FORECASTS_DIR / "latest_forecast.csv"

# === Page Config ===
st.set_page_config(
    page_title="Irish DAM Price Forecast",
    page_icon="⚡",
    layout="wide"
)

# === Caching ===
@st.cache_data(ttl=3600)
def load_training_data():
    """Load the processed training data."""
    if TRAIN_FILE.exists():
        df = pd.read_parquet(TRAIN_FILE)
        return df
    return None


@st.cache_resource
def load_model():
    """Load the trained model."""
    if MODEL_FILE.exists():
        import joblib
        return joblib.load(MODEL_FILE)
    return None


@st.cache_data(ttl=1800)
def load_latest_forecast():
    """Load the latest generated forecast."""
    if FORECAST_FILE.exists():
        return pd.read_csv(FORECAST_FILE, parse_dates=["delivery_time", "generated_at"])
    return None


def get_data_status():
    """Get status of data files."""
    status = {}
    
    for name, path in [
        ("Training Data", TRAIN_FILE),
        ("Model", MODEL_FILE),
        ("Forecast", FORECAST_FILE),
    ]:
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            status[name] = {
                "exists": True,
                "updated": mtime,
                "age_hours": age_hours,
                "status": "🟢 Fresh" if age_hours < 6 else ("🟡 Recent" if age_hours < 24 else "🔴 Stale")
            }
        else:
            status[name] = {"exists": False, "status": "❌ Missing"}
    
    return status


def run_manual_update():
    """Run the daily update script manually."""
    import subprocess
    script_path = PROJECT_ROOT / "scripts" / "daily_update.py"
    
    if script_path.exists():
        result = subprocess.run(
            [sys.executable, str(script_path), "--days", "30"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        return result.returncode == 0, result.stdout + result.stderr
    return False, "Update script not found"


# === Sidebar ===
with st.sidebar:
    st.header("📊 Data Status")
    
    status = get_data_status()
    for name, info in status.items():
        if info.get("exists"):
            st.markdown(f"**{name}**: {info['status']}")
            st.caption(f"Updated: {info['updated'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.markdown(f"**{name}**: {info['status']}")
    
    st.divider()
    
    if st.button("🔄 Run Manual Update", help="Fetch latest data and retrain model"):
        with st.spinner("Running update..."):
            success, output = run_manual_update()
            if success:
                st.success("Update completed!")
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("Update failed")
                with st.expander("See output"):
                    st.code(output)
    
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# === Main Content ===
st.title("⚡ Irish Day-Ahead Power Price Forecast")

# Load data
df = load_training_data()
model = load_model()
forecast = load_latest_forecast()

if df is None:
    st.error("No training data found. Please run the daily update script first.")
    st.code("python scripts/daily_update.py --days 90")
    st.stop()

# Extract target if present
if "target" in df.columns:
    y = df.pop("target")
else:
    y = pd.Series(index=df.index)

# === Summary Metrics ===
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Data Points", f"{len(df):,}")

with col2:
    date_range = f"{df.index.min().date()} → {df.index.max().date()}"
    st.metric("Date Range", date_range)

with col3:
    if "dam_eur_mwh" in df.columns:
        latest_price = df["dam_eur_mwh"].iloc[-1]
        st.metric("Latest Price", f"€{latest_price:.2f}/MWh")

with col4:
    if model is not None:
        st.metric("Model Status", "✅ Loaded")
    else:
        st.metric("Model Status", "❌ Not Found")

st.divider()

# === Date Selection ===
date_min = df.index.min().date()
date_max = df.index.max().date()

selected_date = st.date_input(
    "Select date for forecast:",
    value=date_max,
    min_value=date_min,
    max_value=date_max + timedelta(days=1)
)

# === Forecast Display ===
if model is not None:
    day_data = df[df.index.date == selected_date]
    
    if not day_data.empty:
        # Handle NaN
        day_data_clean = day_data.ffill().bfill().fillna(0)
        predictions = model.predict(day_data_clean)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 Price Forecast - {selected_date}")
            
            fig = go.Figure()
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=day_data.index,
                y=predictions,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8)
            ))
            
            # Actual prices if available
            if "dam_eur_mwh" in day_data.columns:
                actuals = day_data["dam_eur_mwh"]
                if actuals.notna().any():
                    fig.add_trace(go.Scatter(
                        x=day_data.index,
                        y=actuals,
                        mode="lines+markers",
                        name="Actual",
                        line=dict(color="#ff7f0e", width=2, dash="dot"),
                        marker=dict(size=6)
                    ))
            
            fig.update_layout(
                xaxis_title="Hour",
                yaxis_title="Price (€/MWh)",
                hovermode="x unified",
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Statistics")
            
            st.metric("Average", f"€{predictions.mean():.2f}/MWh")
            st.metric("Peak", f"€{predictions.max():.2f}/MWh")
            st.metric("Off-Peak", f"€{predictions.min():.2f}/MWh")
            
            peak_idx = np.argmax(predictions)
            peak_hour = day_data.index[peak_idx].hour
            st.info(f"🕐 Peak hour: {peak_hour}:00")
        
        # Detailed table
        with st.expander("📋 Hourly Details"):
            results = pd.DataFrame({
                "Hour": day_data.index.strftime("%H:%M"),
                "Forecast (€/MWh)": predictions.round(2),
            })
            
            if "dam_eur_mwh" in day_data.columns:
                results["Actual (€/MWh)"] = day_data["dam_eur_mwh"].values.round(2)
                results["Error"] = (predictions - day_data["dam_eur_mwh"].values).round(2)
            
            st.dataframe(results, use_container_width=True, hide_index=True)
            
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
else:
    st.warning("Model not loaded. Please run the daily update script.")

# === Model Performance ===
st.divider()

with st.expander("🔍 Model Performance (Last 7 Days)"):
    if model is not None and len(df) > 24 * 14:
        # Split for backtesting
        split_idx = -24 * 7
        X_test = df.iloc[split_idx:].ffill().bfill().fillna(0)
        y_test = y.iloc[split_idx:]
        
        valid = y_test.notna()
        X_test = X_test[valid]
        y_test = y_test[valid]
        
        if len(X_test) > 0:
            test_pred = model.predict(X_test)
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(y_test, test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            col1, col2 = st.columns(2)
            col1.metric("MAE", f"€{mae:.2f}/MWh")
            col2.metric("RMSE", f"€{rmse:.2f}/MWh")
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test.index,
                y=y_test.values,
                mode="lines",
                name="Actual"
            ))
            fig.add_trace(go.Scatter(
                x=y_test.index,
                y=test_pred,
                mode="lines",
                name="Predicted"
            ))
            fig.update_layout(
                title="Actual vs Predicted (Last 7 Days)",
                xaxis_title="Time",
                yaxis_title="Price (€/MWh)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 14 days of data for performance metrics.")

# === Historical Prices ===
st.divider()

with st.expander("📈 Historical Price Trend"):
    if "dam_eur_mwh" in df.columns:
        # Last 30 days
        recent = df["dam_eur_mwh"].last("30D")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent.index,
            y=recent.values,
            mode="lines",
            name="DAM Price",
            line=dict(color="#2ca02c")
        ))
        
        # Add 24h rolling average
        rolling_avg = recent.rolling(24).mean()
        fig.add_trace(go.Scatter(
            x=rolling_avg.index,
            y=rolling_avg.values,
            mode="lines",
            name="24h Average",
            line=dict(color="#d62728", dash="dash")
        ))
        
        fig.update_layout(
            title="Last 30 Days",
            xaxis_title="Date",
            yaxis_title="Price (€/MWh)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# === Footer ===
st.divider()
st.caption("💡 Data: SEMOpx / ENTSO-E | Weather: Open-Meteo | Automated daily updates")
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
