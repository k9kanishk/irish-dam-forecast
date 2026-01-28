from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .model_factory import ForecastModel


class EnsembleModel(ForecastModel):
    """Ensemble of multiple models with weighted averaging."""

    def __init__(self, models: List[ForecastModel], weights: List[float] | None = None) -> None:
        self.models = models
        self.weights = weights or [1 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
        return predictions

    def predict_interval(
        self,
        X: pd.DataFrame,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_preds = np.array([m.predict(X) for m in self.models])

        point = np.average(all_preds, axis=0, weights=self.weights)
        std = np.std(all_preds, axis=0)

        z = 1.645
        lower = point - z * std
        upper = point + z * std

        return point, lower, upper
