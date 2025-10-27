import pandas as pd


def make_day_ahead_target(dam: pd.Series) -> pd.Series:
    y = dam.shift(-24).rename('y_dam_eur_mwh')
    return y
