#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path

import pandas as pd

from src.features.build_features import build_feature_table
from src.features.targets import make_day_ahead_target

DATA_PATH = Path("data")

if __name__ == '__main__':
    dam = pd.read_csv('data/raw/dam_prices_ie.csv', index_col=0, parse_dates=True).iloc[:,0]
    load_fc = pd.read_csv('data/raw/load_forecast_ie.csv', index_col=0, parse_dates=True).iloc[:,0]
    ws = pd.read_csv('data/raw/wind_solar_forecast_ie.csv', index_col=0, parse_dates=True)
    weather = pd.read_csv('data/raw/weather_hourly.csv', index_col=0, parse_dates=True)

    X = build_feature_table(dam, load_fc, ws, weather)

    # y[t] = price at t+24h (aligned to feature timestamp)
    y = make_day_ahead_target(dam).reindex(X.index)

    df = X.copy()
    df["target"] = y
    df = df.dropna(subset=["target"])

    out_path = DATA_PATH / "processed" / "train.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
