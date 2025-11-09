#!/usr/bin/env python
from __future__ import annotations
import pandas as pd
from src.models.xgb_model import make_model as xgb_factory
from src.models.baseline_lr import make_model as ridge_factory
from src.evaluation.backtest import rolling_origin_cv

if __name__ == '__main__':
    df = pd.read_parquet('data/processed/train.parquet')
    y = df.pop('target')

    print('Ridge:')
    print(rolling_origin_cv(df, y, ridge_factory, n_splits=12))
    print('XGBoost:')
    print(rolling_origin_cv(df, y, xgb_factory, n_splits=12))
