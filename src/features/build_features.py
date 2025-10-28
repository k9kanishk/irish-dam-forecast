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


def build_feature_table(dam: pd.Series,
                        load_fc: pd.Series,
                        windsol_fc: pd.DataFrame,
                        weather: pd.DataFrame) -> pd.DataFrame:

    dam.index = dam.index.tz_localize(None)
    load_fc.index = load_fc.index.tz_localize(None)
    windsol_fc.index = windsol_fc.index.tz_localize(None)
    weather.index = weather.index.tz_localize(None)                        
    df = pd.concat([dam, load_fc, windsol_fc, weather], axis=1).sort_index()
    df['wind_proxy'] = wind_power_proxy(df['wind100m_ms'])
    # lags & rolling stats
    for col in ['dam_eur_mwh','load_forecast_mw','wind_proxy']:
        for L in [1,24,48,72]:
            df[f'{col}_lag{L}'] = df[col].shift(L)
        df[f'{col}_roll24_mean'] = df[col].rolling(24).mean()
        df[f'{col}_roll24_std']  = df[col].rolling(24).std()
    # calendar
    cal = calendar_features(df.index)
    df = df.join(cal)
    return df.dropna()
