import numpy as np
import pandas as pd

from src.models.xgb_quantile import XGBQuantileModel


def training_data():
    np.random.seed(42)
    n = 240
    X = pd.DataFrame(
        {
            "hour": np.tile(np.arange(24), n // 24 + 1)[:n],
            "dow": np.tile(np.arange(7), n // 7 + 1)[:n],
            "load": np.random.uniform(3000, 5000, n),
        }
    )
    y = pd.Series(
        50 + 10 * np.sin(X["hour"] * np.pi / 12) + 0.01 * X["load"] + np.random.normal(0, 5, n)
    )
    return X, y


def test_quantile_model_fit_predict():
    X, y = training_data()
    model = XGBQuantileModel(quantiles=[0.1, 0.5, 0.9], n_estimators=10, max_depth=3)
    model.fit(X, y)

    pred = model.predict(X)
    assert len(pred) == len(X)
    assert not np.any(np.isnan(pred))


def test_quantile_intervals():
    X, y = training_data()
    model = XGBQuantileModel(quantiles=[0.1, 0.5, 0.9], n_estimators=10, max_depth=3)
    model.fit(X, y)

    point, lower, upper = model.predict_interval(X)

    assert np.all(lower <= point)
    assert np.all(point <= upper)
