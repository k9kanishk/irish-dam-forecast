#!/usr/bin/env python
from __future__ import annotations
import pandas as pd
from src.features.build_features import build_feature_table
from src.features.targets import make_day_ahead_target

if __name__ == '__main__':
    dam = pd.read_csv('data/raw/dam_prices_ie.csv', index_col=0, parse_dates=True).iloc[:,0]
    load_fc = pd.read_csv('data/raw/load_forecast_ie.csv', index_col=0, parse_dates=True).iloc[:,0]
    ws = pd.read_csv('data/raw/wind_solar_forecast_ie.csv', index_col=0, parse_dates=True)
    weather = pd.read_csv('data/raw/weather_hourly.csv', index_col=0, parse_dates=True)

    X = build_feature_table(dam, load_fc, ws, weather)
    y = make_day_ahead_target(dam).reindex(X.index)

    df = X.copy(); df['target'] = y
    df.dropna().to_parquet('data/processed/train.parquet')
