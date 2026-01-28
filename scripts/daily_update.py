#!/usr/bin/env python
# scripts/daily_update.py
"""
Daily automated update script for Irish DAM price forecasting.

This script:
1. Fetches latest DAM prices from SEMOpx
2. Fetches weather data
3. Rebuilds the feature dataset
4. Retrains the model
5. Generates tomorrow's forecast
6. Saves results and logs

Schedule this to run daily at 14:00 (after DAM auction results are published ~12:30)

Usage:
    python scripts/daily_update.py
    python scripts/daily_update.py --days 90 --force-refresh
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

# === Configuration ===
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
FORECASTS_DIR = DATA_DIR / "forecasts"

# Create directories
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR, FORECASTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Logging setup ===
LOG_FILE = LOGS_DIR / f"daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def fetch_dam_prices(days: int = 90, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch DAM prices from multiple sources with fallback."""
    logger.info(f"Fetching DAM prices for last {days} days...")
    
    # Try Method 1: SEMOpx scraper
    try:
        from data.semopx_scraper import fetch_recent_dam
        df = fetch_recent_dam(days=days, force_refresh=force_refresh)
        if not df.empty and len(df) > 24:
            logger.info(f"Got {len(df)} rows from SEMOpx scraper")
            return df
    except ImportError:
        logger.warning("SEMOpx scraper not found, trying alternatives...")
    except Exception as e:
        logger.warning(f"SEMOpx scraper failed: {e}")
    
    # Try Method 2: Existing SEMOpx API (HRP files)
    try:
        from data.semopx_api import fetch_dam_hrp_recent
        df = fetch_dam_hrp_recent(days=days, force=force_refresh)
        if not df.empty:
            logger.info(f"Got {len(df)} rows from SEMOpx HRP API")
            return df
    except Exception as e:
        logger.warning(f"SEMOpx HRP API failed: {e}")
    
    # Try Method 3: ENTSO-E
    try:
        from data.entsoe_api import fetch_ie_dam_chunked
        df = fetch_ie_dam_chunked(days=days, force_refresh=force_refresh)
        if not df.empty:
            # Convert to standard format
            df = df.reset_index()
            df.columns = ["ts_utc", "dam_eur_mwh"]
            logger.info(f"Got {len(df)} rows from ENTSO-E")
            return df
    except Exception as e:
        logger.warning(f"ENTSO-E failed: {e}")
    
    # Try Method 4: Load existing Excel files
    try:
        excel_paths = [
            RAW_DIR / "dam_prices.xlsx",
            RAW_DIR / "Lookback2_mkt.xlsx",
            RAW_DIR / "lookback_mkt.xlsx",
            PROJECT_ROOT / "src" / "data" / "raw" / "dam_prices.xlsx",
        ]
        
        for path in excel_paths:
            if path.exists():
                logger.info(f"Loading from {path}")
                df = pd.read_excel(path, engine="openpyxl")
                
                # Try to parse - adjust column names as needed
                cols = {str(c).lower().strip(): c for c in df.columns}
                
                # Find timestamp column
                ts_col = None
                for key in ["timestamp", "datetime", "date", "ts_utc"]:
                    if key in cols:
                        ts_col = cols[key]
                        break
                
                # Find price column
                price_col = None
                for key in ["price_eur", "eur/mwh", "dam_eur_mwh", "price"]:
                    if key in cols:
                        price_col = cols[key]
                        break
                
                if ts_col and price_col:
                    out = pd.DataFrame({
                        "ts_utc": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
                        "dam_eur_mwh": pd.to_numeric(df[price_col], errors="coerce")
                    })
                    out = out.dropna().sort_values("ts_utc")
                    
                    # Filter to requested days
                    cutoff = datetime.now(tz=pd.Timestamp.now().tz) - timedelta(days=days)
                    out = out[out["ts_utc"] >= cutoff]
                    
                    if len(out) > 24:
                        logger.info(f"Got {len(out)} rows from Excel file")
                        return out
                        
    except Exception as e:
        logger.warning(f"Excel loading failed: {e}")
    
    logger.error("All DAM price sources failed!")
    return pd.DataFrame(columns=["ts_utc", "dam_eur_mwh"])


def fetch_weather_data(start_date: str, end_date: str, lat: float = 53.4, lon: float = -8.2) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo."""
    logger.info(f"Fetching weather data from {start_date} to {end_date}...")
    
    try:
        from data.weather import fetch_hourly
        df = fetch_hourly(lat, lon, start_date, end_date)
        logger.info(f"Got {len(df)} weather records")
        return df
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return pd.DataFrame()


def fetch_load_forecast(start_date: str, end_date: str) -> pd.Series:
    """Fetch load forecast from ENTSO-E."""
    logger.info(f"Fetching load forecast...")
    
    try:
        from data.entsoe_api import Entsoe
        e = Entsoe()
        load_fc = e.load_forecast(start_date, end_date)
        logger.info(f"Got {len(load_fc)} load forecast records")
        return load_fc
    except Exception as e:
        logger.warning(f"Load forecast failed: {e}")
        return pd.Series(dtype=float, name="load_forecast_mw")


def fetch_wind_solar_forecast(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch wind/solar forecast from ENTSO-E."""
    logger.info(f"Fetching wind/solar forecast...")
    
    try:
        from data.entsoe_api import Entsoe
        e = Entsoe()
        ws_fc = e.wind_solar_forecast(start_date, end_date)
        logger.info(f"Got {len(ws_fc)} wind/solar records")
        return ws_fc
    except Exception as e:
        logger.warning(f"Wind/solar forecast failed: {e}")
        return pd.DataFrame()


def build_features(dam: pd.Series, load_fc: pd.Series, ws_fc: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Build feature table."""
    logger.info("Building features...")
    
    try:
        from features.build_features import build_feature_table
        X = build_feature_table(dam, load_fc, ws_fc, weather)
        logger.info(f"Built {len(X)} feature rows with {len(X.columns)} columns")
        return X
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        # Minimal fallback features
        idx = dam.index
        X = pd.DataFrame({
            "hour": idx.hour,
            "dow": idx.dayofweek,
            "month": idx.month,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "dam_eur_mwh": dam.values,
        }, index=idx)
        return X


def build_target(dam: pd.Series, horizon_hours: int = 24) -> pd.Series:
    """Build prediction target."""
    from features.targets import make_day_ahead_target
    return make_day_ahead_target(dam, horizon_hours=horizon_hours)


def train_model(X: pd.DataFrame, y: pd.Series):
    """Train the forecasting model."""
    logger.info(f"Training model on {len(X)} samples...")
    
    try:
        from models.xgb_model import make_model
        model = make_model()
    except ImportError:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            tree_method="hist",
            random_state=42
        )
    
    # Filter valid rows
    valid = y.notna() & np.isfinite(y)
    X_train = X[valid]
    y_train = y[valid]
    
    # Handle any remaining NaN in features
    X_train = X_train.ffill().bfill().fillna(0)
    
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    
    return model


def generate_forecast(model, X: pd.DataFrame, forecast_date: datetime.date) -> pd.DataFrame:
    """Generate forecast for a specific date."""
    logger.info(f"Generating forecast for {forecast_date}...")
    
    # Filter to forecast date
    day_data = X[X.index.date == forecast_date]
    
    if day_data.empty:
        logger.warning(f"No data for {forecast_date}")
        return pd.DataFrame()
    
    # Handle NaN
    day_data = day_data.ffill().bfill().fillna(0)
    
    # Predict
    predictions = model.predict(day_data)
    
    results = pd.DataFrame({
        "delivery_time": day_data.index,
        "forecast_eur_mwh": predictions,
        "generated_at": datetime.now(),
    })
    
    logger.info(f"Generated {len(results)} hourly forecasts")
    logger.info(f"  Min: €{predictions.min():.2f}/MWh")
    logger.info(f"  Max: €{predictions.max():.2f}/MWh")
    logger.info(f"  Mean: €{predictions.mean():.2f}/MWh")
    
    return results


def save_model(model, path: Path):
    """Save the trained model."""
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: Path):
    """Load a trained model."""
    import joblib
    return joblib.load(path)


def save_forecast(forecast: pd.DataFrame, forecast_date: datetime.date):
    """Save forecast to file."""
    filename = FORECASTS_DIR / f"forecast_{forecast_date.strftime('%Y%m%d')}.csv"
    forecast.to_csv(filename, index=False)
    logger.info(f"Forecast saved to {filename}")
    
    # Also save latest
    latest_file = FORECASTS_DIR / "latest_forecast.csv"
    forecast.to_csv(latest_file, index=False)


def main(days: int = 90, force_refresh: bool = False):
    """Main daily update routine."""
    logger.info("=" * 60)
    logger.info("Starting daily update")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Fetch DAM prices
        dam_df = fetch_dam_prices(days=days, force_refresh=force_refresh)
        if dam_df.empty:
            logger.error("No DAM prices available. Aborting.")
            return False
        
        # Ensure UTC timezone
        dam_df["ts_utc"] = pd.to_datetime(dam_df["ts_utc"], utc=True)
        
        # Save raw data
        dam_df.to_parquet(RAW_DIR / "dam_prices_latest.parquet", index=False)
        
        # Convert to Dublin timezone for features (tz-naive)
        dam_series = dam_df.set_index("ts_utc")["dam_eur_mwh"]
        dam_series = dam_series.tz_convert("Europe/Dublin").tz_localize(None)
        dam_series = dam_series[~dam_series.index.duplicated(keep="last")]
        
        # Step 2: Get date range
        start_date = dam_series.index.min().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        
        # Step 3: Fetch supporting data
        weather = fetch_weather_data(start_date, end_date)
        load_fc = fetch_load_forecast(start_date, end_date)
        ws_fc = fetch_wind_solar_forecast(start_date, end_date)
        
        # Align timezones (Dublin tz-naive)
        def to_dublin_naive(x):
            if not hasattr(x, "index") or not isinstance(x.index, pd.DatetimeIndex):
                return x
            idx = x.index
            if idx.tz is not None:
                idx = idx.tz_convert("Europe/Dublin").tz_localize(None)
            else:
                idx = idx.tz_localize("UTC").tz_convert("Europe/Dublin").tz_localize(None)
            x.index = idx
            return x
        
        load_fc = to_dublin_naive(load_fc)
        ws_fc = to_dublin_naive(ws_fc)
        weather = to_dublin_naive(weather)
        
        # Deduplicate
        dam_series = dam_series[~dam_series.index.duplicated(keep="last")]
        if hasattr(load_fc, "index"):
            load_fc = load_fc[~load_fc.index.duplicated(keep="last")]
        if hasattr(ws_fc, "index"):
            ws_fc = ws_fc[~ws_fc.index.duplicated(keep="last")]
        if hasattr(weather, "index"):
            weather = weather[~weather.index.duplicated(keep="last")]
        
        # Step 4: Build features
        X = build_features(dam_series, load_fc, ws_fc, weather)
        y = build_target(dam_series)
        
        # Align X and y
        common_idx = X.index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Step 5: Save processed dataset
        train_df = X.copy()
        train_df["target"] = y
        train_df.to_parquet(PROCESSED_DIR / "train.parquet")
        logger.info(f"Saved training data: {len(train_df)} rows")
        
        # Step 6: Train model
        model = train_model(X, y)
        
        # Save model
        model_path = MODELS_DIR / f"xgb_model_{datetime.now().strftime('%Y%m%d')}.joblib"
        save_model(model, model_path)
        save_model(model, MODELS_DIR / "latest_model.joblib")
        
        # Step 7: Generate forecasts
        tomorrow = datetime.now().date() + timedelta(days=1)
        
        # We need features for tomorrow - use last available + time features
        if tomorrow not in [d.date() for d in X.index]:
            # Create synthetic row for tomorrow using last available data
            logger.info("Creating feature estimates for tomorrow...")
            
            last_row = X.iloc[-1:].copy()
            new_idx = pd.date_range(
                start=datetime.combine(tomorrow, datetime.min.time()),
                periods=24,
                freq="H"
            )
            
            tomorrow_features = pd.DataFrame(index=new_idx)
            for col in X.columns:
                if col in ["hour", "dow", "month", "is_weekend", "is_peak"]:
                    # Calendar features - compute fresh
                    if col == "hour":
                        tomorrow_features[col] = new_idx.hour
                    elif col == "dow":
                        tomorrow_features[col] = new_idx.dayofweek
                    elif col == "month":
                        tomorrow_features[col] = new_idx.month
                    elif col == "is_weekend":
                        tomorrow_features[col] = (new_idx.dayofweek >= 5).astype(int)
                    elif col == "is_peak":
                        tomorrow_features[col] = new_idx.hour.isin(range(7, 23)).astype(int)
                else:
                    # Use last known value for other features
                    tomorrow_features[col] = last_row[col].values[0]
            
            X_forecast = pd.concat([X, tomorrow_features])
        else:
            X_forecast = X
        
        forecast = generate_forecast(model, X_forecast, tomorrow)
        if not forecast.empty:
            save_forecast(forecast, tomorrow)
        
        logger.info("=" * 60)
        logger.info("Daily update completed successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Daily update failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily DAM price forecast update")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh all cached data")
    
    args = parser.parse_args()
    
    success = main(days=args.days, force_refresh=args.force_refresh)
    sys.exit(0 if success else 1)
