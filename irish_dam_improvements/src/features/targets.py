# src/features/targets.py
"""
Target variable creation for Irish DAM price forecasting.
"""
import pandas as pd


def make_day_ahead_target(dam: pd.Series) -> pd.Series:
    """
    Create target variable for day-ahead forecasting.
    
    The target at time t is the price at time t+24h.
    
    Args:
        dam: DAM prices series
        
    Returns:
        Target series (shifted by -24 hours)
    """
    y = dam.shift(-24).rename("y_dam_eur_mwh")
    return y
