from __future__ import annotations
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from .metrics import rmse, mape


def rolling_origin_cv(X: pd.DataFrame, y: pd.Series, model_factory, n_splits=12):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        mdl = model_factory()
        mdl.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = mdl.predict(X.iloc[test_idx])
        rows.append({
            'fold': i+1,
            'rmse': rmse(y.iloc[test_idx].values, pred),
            'mape': mape(y.iloc[test_idx].values, pred),
            'n_test': len(test_idx)
        })
    return pd.DataFrame(rows)
