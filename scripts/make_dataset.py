#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path

import yaml

import pandas as pd

from src.features.build_features import build_feature_table
from src.features.targets import make_day_ahead_target

DEFAULT_DATA_PATH = Path("data")
CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r") as handle:
        return yaml.safe_load(handle) or {}


def resolve_processed_path(cfg: dict) -> Path:
    processed = cfg.get("paths", {}).get("processed")
    return Path(processed) if processed else (DEFAULT_DATA_PATH / "processed")

if __name__ == '__main__':
    cfg = load_config()
    processed_dir = resolve_processed_path(cfg)
    horizon_hours = cfg.get("train", {}).get("horizon_hours", 24)

    dam = pd.read_csv("data/raw/dam_prices_ie.csv", index_col=0, parse_dates=True).iloc[:, 0]
    load_fc = pd.read_csv("data/raw/load_forecast_ie.csv", index_col=0, parse_dates=True).iloc[:, 0]
    ws = pd.read_csv("data/raw/wind_solar_forecast_ie.csv", index_col=0, parse_dates=True)
    weather = pd.read_csv("data/raw/weather_hourly.csv", index_col=0, parse_dates=True)

    X = build_feature_table(dam, load_fc, ws, weather)

    # y[t] = price at t+24h (aligned to feature timestamp)
    y = make_day_ahead_target(dam, horizon_hours=horizon_hours).reindex(X.index)

    df = X.copy()
    df["target"] = y
    df = df.dropna(subset=["target"])

    out_path = processed_dir / "train.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
