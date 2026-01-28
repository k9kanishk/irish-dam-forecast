from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .model_factory import ForecastModel


class XGBQuantileModel(ForecastModel):
    """XGBoost quantile regression for probabilistic forecasts."""

    def __init__(self, quantiles: list[float] | None = None, **xgb_params) -> None:
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.xgb_params = {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42,
            **xgb_params,
        }
        self.models: dict[float, xgb.XGBRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBQuantileModel":
        for q in self.quantiles:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                **self.xgb_params,
            )
            model.fit(X, y)
            self.models[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.models[0.5].predict(X)

    def predict_interval(
        self,
        X: pd.DataFrame,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        lower_model = self.models[min(self.quantiles, key=lambda x: abs(x - lower_q))]
        upper_model = self.models[min(self.quantiles, key=lambda x: abs(x - upper_q))]

        point = self.predict(X)
        lower = lower_model.predict(X)
        upper = upper_model.predict(X)

        return point, lower, upper

    def predict_all_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        results = {}
        for q, model in self.models.items():
            results[f"q{int(q * 100)}"] = model.predict(X)
        return pd.DataFrame(results, index=X.index)
