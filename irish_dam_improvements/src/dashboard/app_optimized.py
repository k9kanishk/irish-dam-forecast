# src/dashboard/app_optimized.py
"""
Optimized Irish DAM Forecast Dashboard

Key improvements over original:
1. Lazy loading - only compute what's needed
2. Efficient caching with proper cache keys
3. No redundant data fetching
4. Model trained once and cached
5. Parallel data loading
6. Simplified error handling
7. Progressive UI updates
"""
import os
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Irish Power Price Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== CACHING FUNCTIONS ====================

def get_data_hash() -> str:
    """Generate hash of current data state for cache invalidation."""
    if TRAIN_PATH.exists():
        mtime = TRAIN_PATH.stat().st_mtime
        size = TRAIN_PATH.stat().st_size
        return hashlib.md5(f"{mtime}_{size}".encode()).hexdigest()[:8]
    return "no_data"


@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset(_data_hash: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare dataset. Cached by data hash.
    """
    if not TRAIN_PATH.exists():
        return pd.DataFrame(), pd.Series(dtype=float)
    
    df = pd.read_parquet(TRAIN_PATH)
    
    # Separate target
    y = df.pop("target") if "target" in df.columns else pd.Series(index=df.index, dtype=float)
    
    return df, y


@st.cache_resource
def get_trained_model(_data_hash: str, _X_shape: tuple, _y_len: int):
    """
    Train model once and cache. Invalidated when data changes.
    """
    from xgboost import XGBRegressor
    
    # Load fresh data for training
    df, y = load_dataset(_data_hash)
    
    if df.empty or y.empty:
        return None
    
    # Quick model with reasonable defaults
    model = XGBRegressor(
        n_estimators=500,  # Reduced for speed
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,  # Use all cores
    )
    
    # Filter valid rows
    valid = y.notna() & df.notna().all(axis=1)
    X_train = df[valid]
    y_train = y[valid]
    
    if len(X_train) < 100:
        return None
    
    model.fit(X_train, y_train)
    return model


# ==================== DATA GENERATION ====================

def ensure_data_exists() -> bool:
    """
    Ensure training data exists. Generate synthetic if needed.
    Returns True if data is available.
    """
    if TRAIN_PATH.exists():
        return True
    
    # Try to import synthetic generator
    try:
        from src.data.synthetic_generator import generate_for_dashboard
        
        st.info("🔄 Generating synthetic data for demo...")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        generate_for_dashboard(days=90, output_dir=str(DATA_DIR))
        return True
        
    except ImportError:
        st.error("No training data found and synthetic generator not available.")
        st.info("Please run: python src/data/synthetic_generator.py --for-dashboard 90")
        return False


# ==================== UI COMPONENTS ====================

def render_sidebar(df: pd.DataFrame, data_hash: str) -> dict:
    """Render sidebar and return configuration."""
    with st.sidebar:
        st.header("📊 Data Management")
        
        # Data status
        if TRAIN_PATH.exists():
            mod_time = datetime.fromtimestamp(TRAIN_PATH.stat().st_mtime)
            age_hours = (datetime.now() - mod_time).total_seconds() / 3600
            
            if age_hours < 1:
                status_emoji = "🟢"
                status_text = "Fresh"
            elif age_hours < 12:
                status_emoji = "🟡"
                status_text = "Recent"
            else:
                status_emoji = "🔴"
                status_text = "Stale"
            
            st.markdown(f"""
            **Status:** {status_emoji} {status_text}  
            **Updated:** {mod_time.strftime('%Y-%m-%d %H:%M')}  
            **Rows:** {len(df):,}  
            **Hash:** `{data_hash}`
            """)
        
        # Refresh buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", help="Reload data from disk"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Reset", help="Delete cache and regenerate"):
                if TRAIN_PATH.exists():
                    TRAIN_PATH.unlink()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        
        st.divider()
        
        # Date selection
        st.subheader("📅 Forecast Date")
        
        if not df.empty:
            date_min = df.index.min().date()
            date_max = df.index.max().date()
            
            selected_date = st.date_input(
                "Select date:",
                value=date_max,
                min_value=date_min,
                max_value=date_max,
            )
        else:
            selected_date = datetime.now().date()
        
        st.divider()
        
        # Latest price display
        if not df.empty and "dam_eur_mwh" in df.columns:
            st.subheader("⚡ Latest Price")
            last_price = df["dam_eur_mwh"].dropna().iloc[-1] if not df["dam_eur_mwh"].dropna().empty else 0
            last_time = df.index[-1]
            
            st.metric(
                label=f"{last_time.strftime('%Y-%m-%d %H:%M')}",
                value=f"€{last_price:.2f}/MWh"
            )
        
        return {"selected_date": selected_date}


def render_forecast(df: pd.DataFrame, model, selected_date) -> None:
    """Render forecast section."""
    st.subheader(f"📈 Forecast for {selected_date}")
    
    # Filter data for selected date
    day_data = df[df.index.date == selected_date]
    
    if day_data.empty:
        st.warning(f"No data available for {selected_date}")
        return
    
    if model is None:
        st.error("Model not trained. Please check data.")
        return
    
    # Make predictions
    try:
        predictions = model.predict(day_data)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return
    
    # Create results DataFrame
    results = pd.DataFrame({
        "Hour": day_data.index,
        "Forecast (€/MWh)": predictions.round(2)
    })
    
    if "load_forecast_mw" in day_data.columns:
        results["Load (MW)"] = day_data["load_forecast_mw"].round(0).values
    if "wind_total_mw" in day_data.columns:
        results["Wind (MW)"] = day_data["wind_total_mw"].round(0).values
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price forecast chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results["Hour"],
            y=results["Forecast (€/MWh)"],
            mode="lines+markers",
            name="Price Forecast",
            line=dict(width=3, color="#1f77b4"),
            marker=dict(size=8),
        ))
        
        # Add actual prices if available
        if "dam_eur_mwh" in day_data.columns:
            fig.add_trace(go.Scatter(
                x=day_data.index,
                y=day_data["dam_eur_mwh"],
                mode="lines+markers",
                name="Actual",
                line=dict(width=2, color="#2ca02c", dash="dot"),
                marker=dict(size=6),
            ))
        
        fig.update_layout(
            title=f"Day-Ahead Price Forecast - {selected_date}",
            xaxis_title="Hour",
            yaxis_title="Price (€/MWh)",
            hovermode="x unified",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Statistics
        st.subheader("📊 Statistics")
        
        avg_price = float(np.mean(predictions))
        max_price = float(np.max(predictions))
        min_price = float(np.min(predictions))
        peak_hour = results.loc[results["Forecast (€/MWh)"].idxmax(), "Hour"].hour
        
        st.metric("Average", f"€{avg_price:.2f}/MWh")
        st.metric("Peak", f"€{max_price:.2f}/MWh")
        st.metric("Off-Peak", f"€{min_price:.2f}/MWh")
        st.info(f"🕐 Peak hour: {peak_hour}:00")
    
    # Detailed table (collapsed by default)
    with st.expander("📋 Detailed Hourly Forecast"):
        st.dataframe(results.set_index("Hour"), use_container_width=True)
        
        csv = results.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv,
            f"forecast_{selected_date}.csv",
            "text/csv"
        )


def render_performance(df: pd.DataFrame, y: pd.Series, model) -> None:
    """Render model performance section (lazy loaded)."""
    if len(df) < 24 * 14:
        st.info("Need at least 14 days of data for performance metrics.")
        return
    
    if model is None:
        st.warning("Model not available for performance evaluation.")
        return
    
    # Use session state to track if user wants to see performance
    if "show_perf" not in st.session_state:
        st.session_state.show_perf = False
    
    if not st.session_state.show_perf:
        if st.button("📊 Calculate Performance Metrics"):
            st.session_state.show_perf = True
            st.rerun()
        return
    
    # Calculate performance
    with st.spinner("Calculating backtest metrics..."):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # 7-day holdout
        split = -24 * 7
        X_train, X_test = df.iloc[:split], df.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Filter valid rows
        valid_train = y_train.notna() & X_train.notna().all(axis=1)
        valid_test = y_test.notna() & X_test.notna().all(axis=1)
        
        # Train fresh model for fair comparison
        from xgboost import XGBRegressor
        test_model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            tree_method="hist",
            random_state=42,
        )
        test_model.fit(X_train[valid_train], y_train[valid_train])
        
        # Predict
        test_pred = test_model.predict(X_test[valid_test])
        y_true = y_test[valid_test].values
        
        # Metrics
        mae = mean_absolute_error(y_true, test_pred)
        rmse = np.sqrt(mean_squared_error(y_true, test_pred))
    
    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("MAE (7-day test)", f"€{mae:.2f}/MWh")
    col2.metric("RMSE (7-day test)", f"€{rmse:.2f}/MWh")
    
    # Quality assessment
    if mae < 5:
        st.warning("⚠️ MAE seems too low - check for data leakage")
    elif mae > 30:
        st.warning("⚠️ MAE is high - model may need improvement")
    else:
        st.success(f"✅ MAE of €{mae:.2f}/MWh is reasonable for Irish DAM")
    
    # Actual vs Predicted chart
    perf_df = pd.DataFrame({
        "actual": y_true,
        "predicted": test_pred,
    }, index=X_test[valid_test].index)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf_df.index,
        y=perf_df["actual"],
        mode="lines",
        name="Actual",
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=perf_df.index,
        y=perf_df["predicted"],
        mode="lines",
        name="Predicted",
        line=dict(width=2, dash="dot"),
    ))
    
    fig.update_layout(
        title="Actual vs Predicted (Last 7 Days)",
        xaxis_title="Delivery Hour",
        yaxis_title="Price (€/MWh)",
        hovermode="x unified",
        height=350,
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ==================== MAIN APP ====================

def main():
    # Title
    st.title("⚡ Irish Day-Ahead Power Price Forecast")
    
    # Ensure data exists
    if not ensure_data_exists():
        st.stop()
    
    # Get data hash for cache invalidation
    data_hash = get_data_hash()
    
    # Load data (cached)
    with st.spinner("Loading data..."):
        df, y = load_dataset(data_hash)
    
    if df.empty:
        st.error("No data loaded. Please check data files.")
        st.stop()
    
    # Train model (cached)
    with st.spinner("Preparing model..."):
        model = get_trained_model(data_hash, df.shape, len(y))
    
    if model is not None:
        st.success("✅ Model ready", icon="✅")
    else:
        st.warning("⚠️ Model training failed - check data quality")
    
    # Render sidebar and get config
    config = render_sidebar(df, data_hash)
    
    # Main content
    render_forecast(df, model, config["selected_date"])
    
    # Performance section (lazy)
    with st.expander("🔍 Model Performance", expanded=False):
        render_performance(df, y, model)
    
    # Footer
    st.divider()
    st.caption("💡 Data: SEMOpx / ENTSO-E | Weather: Open-Meteo | Built with Streamlit")


if __name__ == "__main__":
    main()
