#!/usr/bin/env python
from __future__ import annotations
import pandas as pd
from src.evaluation.metrics import rmse, mape

if __name__ == '__main__':
    df = pd.read_parquet('data/processed/train.parquet')
    y = df.pop('target')
    split = -24*30
    mdl = __import__('src.models.xgb_model', fromlist=['make_model']).make_model()
    mdl.fit(df.iloc[:split], y.iloc[:split])
    pred = mdl.predict(df.iloc[split:])
    print({'RMSE': rmse(y.iloc[split:].values, pred), 'MAPE_%': mape(y.iloc[split:].values, pred)})
