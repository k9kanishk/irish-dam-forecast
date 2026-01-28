from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .metrics import mae, mape, rmse


def rolling_origin_cv(X: pd.DataFrame, y: pd.Series, model_factory, n_splits=12):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        mdl = model_factory()
        mdl.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = mdl.predict(X.iloc[test_idx])
        rows.append(
            {
                "fold": i + 1,
                "rmse": rmse(y.iloc[test_idx].values, pred),
                "mape": mape(y.iloc[test_idx].values, pred),
                "n_test": len(test_idx),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict
    fold_metrics: pd.DataFrame
    residuals: pd.Series


class TimeSeriesBacktester:
    """Walk-forward backtesting for electricity price forecasting."""

    def __init__(
        self,
        model_factory: Callable,
        min_train_days: int = 90,
        test_days: int = 7,
        step_days: int = 7,
        retrain_every: int = 7,
    ) -> None:
        self.model_factory = model_factory
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.step_days = step_days
        self.retrain_every = retrain_every

    def run(self, X: pd.DataFrame, y: pd.Series) -> BacktestResult:
        results = []
        fold_metrics = []

        min_train_hours = self.min_train_days * 24
        test_hours = self.test_days * 24
        step_hours = self.step_days * 24

        model = None
        last_train_end = None

        for test_start in range(min_train_hours, len(X) - test_hours, step_hours):
            test_end = test_start + test_hours

            if model is None or (
                last_train_end is not None
                and test_start - last_train_end >= self.retrain_every * 24
            ):
                X_train = X.iloc[:test_start]
                y_train = y.iloc[:test_start]

                model = self.model_factory()
                model.fit(X_train, y_train)
                last_train_end = test_start

            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            pred = model.predict(X_test)

            fold_result = pd.DataFrame(
                {
                    "actual": y_test.values,
                    "predicted": pred,
                    "fold": len(fold_metrics),
                },
                index=y_test.index,
            )
            results.append(fold_result)

            fold_metrics.append(
                {
                    "fold": len(fold_metrics),
                    "start": X_test.index[0],
                    "end": X_test.index[-1],
                    "mae": mae(y_test.values, pred),
                    "rmse": rmse(y_test.values, pred),
                    "mape": mape(y_test.values, pred),
                    "n_samples": len(y_test),
                }
            )

        all_results = pd.concat(results)
        residuals = all_results["actual"] - all_results["predicted"]

        overall_metrics = {
            "mae": residuals.abs().mean(),
            "rmse": np.sqrt((residuals**2).mean()),
            "mape": (residuals.abs() / all_results["actual"].clip(lower=1)).mean() * 100,
            "bias": residuals.mean(),
            "n_folds": len(fold_metrics),
            "total_samples": len(all_results),
        }

        return BacktestResult(
            predictions=all_results,
            metrics=overall_metrics,
            fold_metrics=pd.DataFrame(fold_metrics),
            residuals=residuals,
        )
