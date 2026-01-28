from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .model_factory import ForecastModel


class LightGBMModel(ForecastModel):
    """LightGBM regressor with defaults for time series."""

    def __init__(self, **params) -> None:
        self.params = {
            "n_estimators": 1000,
            "max_depth": 8,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": -1,
            **params,
        }
        self.model: lgb.LGBMRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)
