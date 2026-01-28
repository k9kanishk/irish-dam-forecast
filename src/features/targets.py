import pandas as pd


def make_day_ahead_target(dam: pd.Series, horizon_hours: int = 24) -> pd.Series:
    """Create a day-ahead target aligned to the feature timestamp.

    y[t] corresponds to dam[t + horizon_hours], so features at time t predict
    the delivery price at t + horizon_hours without shifting the index.
    """
    y = dam.shift(-horizon_hours).rename("y_dam_eur_mwh")
    return y
