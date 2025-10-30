from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.holidays_ie import is_ie_holiday

CALENDAR_COLS = ['hour','dow','is_weekend','is_holiday','month','quarter']


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df['hour'] = idx.hour
    df['dow'] = idx.dayofweek
    df['is_weekend'] = (df['dow']>=5).astype(int)
    df['is_holiday'] = [int(is_ie_holiday(d.date())) for d in idx]
    df['month'] = idx.month
    df['quarter'] = idx.quarter
    return df


# Convert wind speed to rough power proxy and % of demand

def wind_power_proxy(wind_ms: pd.Series) -> pd.Series:
    v = wind_ms.clip(lower=3, upper=25)
    proxy = ((v-3)/(25-3))**3
    return proxy.rename('wind_proxy')


import pandas as pd
from .utils import wind_power_proxy  # already used in your file

def build_feature_table(
    dam: pd.Series,
    load_fc: pd.Series,
    windsol_fc: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align all inputs on the **intersection** of timestamps (hourly),
    then build lags/rolls and calendar features.
    Always returns a DataFrame (may be empty if no overlap).
    """

    # Ensure expected names
    dam = dam.copy(); dam.name = "dam_eur_mwh"
    load_fc = load_fc.copy(); load_fc.name = "load_forecast_mw"
    windsol_fc = windsol_fc.copy()
    weather = weather.copy()

    # Pick useful wind/solar cols if present
    keep_ws = [c for c in ["wind_total_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw"]
               if c in getattr(windsol_fc, "columns", [])]
    if keep_ws:
        windsol_fc = windsol_fc[keep_ws]
    else:
        # empty frame with correct index will be set after reindex
        windsol_fc = pd.DataFrame(index=dam.index)

    # ---- Align to common hourly index (intersection) ----
    idx = dam.index
    idx = idx.intersection(load_fc.index)
    if not windsol_fc.empty:
        idx = idx.intersection(windsol_fc.index)
    if not weather.empty:
        idx = idx.intersection(weather.index)
    idx = idx.sort_values()

    if len(idx) == 0:
        return pd.DataFrame()  # caller will handle

    dam = dam.reindex(idx)
    load_fc = load_fc.reindex(idx)
    windsol_fc = windsol_fc.reindex(idx)
    weather = weather.reindex(idx)

    # Concatenate AFTER alignment (no NaNs solely due to mismatched stamps)
    df = pd.concat([dam, load_fc, windsol_fc, weather], axis=1)

    # Wind proxy from weather if present; otherwise fallback to wind power cols
    if "wind100m_ms" in df.columns:
        df["wind_proxy"] = wind_power_proxy(df["wind100m_ms"])
    elif keep_ws:
        df["wind_proxy"] = windsol_fc[keep_ws].sum(axis=1)
    else:
        df["wind_proxy"] = pd.NA


      # --- NEW: forecast wind share of demand ---
    wind_cols = [c for c in df.columns if "wind" in str(c).lower()]
    if wind_cols:
        df["wind_mw_fc"] = df[wind_cols].sum(axis=1)
        df["wind_share_fc"] = (df["wind_mw_fc"] / df["load_forecast_mw"].clip(lower=1)).clip(0, 1)

    
    # Lags & rolling stats (tolerate partial windows)
    for col in ["dam_eur_mwh", "load_forecast_mw", "wind_proxy"]:
        if col in df.columns:
            for L in [1, 24, 48, 72]:
                df[f"{col}_lag{L}"] = df[col].shift(L)
            df[f"{col}_roll24_mean"] = df[col].rolling(24, min_periods=12).mean()
            df[f"{col}_roll24_std"]  = df[col].rolling(24, min_periods=12).std()

    # Calendar features
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month

    return df
