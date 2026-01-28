import numpy as np
import pandas as pd

from src.features.advanced_features import AdvancedFeatureBuilder


def sample_data() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=168, freq="H")
    return pd.DataFrame(
        {
            "dam_eur_mwh": np.random.uniform(50, 150, 168),
            "load_forecast_mw": np.random.uniform(3000, 5000, 168),
            "wind_total_mw": np.random.uniform(500, 2000, 168),
            "temperature_2m": np.random.uniform(5, 20, 168),
        },
        index=idx,
    )


def test_calendar_features():
    data = sample_data()
    builder = AdvancedFeatureBuilder(data)
    result = builder.add_calendar_features().build()

    assert "hour" in result.columns
    assert "dow" in result.columns
    assert "is_weekend" in result.columns
    assert result["hour"].max() == 23
    assert result["hour"].min() == 0


def test_price_features():
    data = sample_data()
    builder = AdvancedFeatureBuilder(data)
    result = builder.add_price_features().build()

    assert "price_lag_24h" in result.columns
    assert "price_roll_24h_mean" in result.columns
    assert result["price_lag_24h"].iloc[:24].isna().all()


def test_no_future_leakage():
    data = sample_data()
    builder = AdvancedFeatureBuilder(data)
    result = builder.add_price_features().build()

    for col in result.columns:
        if "lag" in col or "roll" in col:
            assert result[col].iloc[0] != result["dam_eur_mwh"].iloc[0] or pd.isna(
                result[col].iloc[0]
            )
