import numpy as np
import pandas as pd

from src.data.validators import PriceDataValidator


def test_outlier_detection():
    idx = pd.date_range("2024-01-01", periods=100, freq="H")
    prices = pd.Series(np.random.uniform(50, 100, 100), index=idx)
    prices.iloc[50] = 5000

    df = pd.DataFrame({"dam_eur_mwh": prices})
    validator = PriceDataValidator(df)
    outliers = validator.check_outliers()

    assert outliers.iloc[50]
    assert outliers.sum() >= 1
