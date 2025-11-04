# PATCHED src/features/build_features.py
# This version ensures dam_eur_mwh and load_forecast_mw are included in output

from __future__ import annotations
import pandas as pd
import numpy as np

def wind_power_proxy(wind_ms: pd.Series) -> pd.Series:
    """Convert wind speed to power proxy"""
    v = wind_ms.clip(lower=3, upper=25)
    proxy = ((v-3)/(25-3))**3
    return proxy.rename('wind_proxy')

def build_feature_table(
    dam: pd.Series,
    load_fc: pd.Series,
    windsol_fc: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature table with all necessary columns included.
    """
    # Ensure expected names
    dam = dam.copy()
    dam.name = "dam_eur_mwh"
    
    load_fc = load_fc.copy()
    load_fc.name = "load_forecast_mw"
    
    # Start with base columns
    df = pd.DataFrame(index=dam.index)
    
    # Add the raw data columns (CRITICAL - these were missing!)
    df['dam_eur_mwh'] = dam
    df['load_forecast_mw'] = load_fc
    
    # Add wind/solar if available
    if isinstance(windsol_fc, pd.DataFrame) and not windsol_fc.empty:
        keep_cols = [c for c in windsol_fc.columns 
                     if any(x in str(c).lower() for x in ['wind', 'solar'])]

        for col in keep_cols:
            df[col] = windsol_fc[col]
    
    # Add weather if available
    if isinstance(weather, pd.DataFrame) and not weather.empty:
        for col in weather.columns:
            df[col] = weather[col]
    
    # Create wind proxy
    if 'wind100m_ms' in df.columns:
        df['wind_proxy'] = wind_power_proxy(df['wind100m_ms'])
    elif 'wind_total_mw' in df.columns:
        df['wind_proxy'] = df['wind_total_mw'] / 2000  # Normalize
    
    # Add lags for key features
    for col in ['dam_eur_mwh', 'load_forecast_mw']:
        if col in df.columns:
            # Lags
            for lag in [1, 24, 48, 72]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
            # Rolling statistics
            df[f'{col}_roll24_mean'] = df[col].rolling(24, min_periods=12).mean()
            df[f'{col}_roll24_std'] = df[col].rolling(24, min_periods=12).std()
    
    # Add calendar features
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    
    # Peak hours indicator
    df['is_peak'] = df['hour'].between(7, 22).astype(int)
    
    # Wind share if we have both wind and load
    if 'wind_total_mw' in df.columns and 'load_forecast_mw' in df.columns:
        df['wind_share'] = (df['wind_total_mw'] / df['load_forecast_mw'].clip(lower=1)).clip(0, 1)
    
    return df


# Also update the targets.py if needed
def make_day_ahead_target(dam: pd.Series) -> pd.Series:
    """Create target variable - next day prices"""
    y = dam.shift(-24).rename('y_dam_eur_mwh')
    return y
