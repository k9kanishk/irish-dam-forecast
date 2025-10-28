# src/features/utils.py
import pandas as pd

def wind_power_proxy(wind_ms: pd.Series) -> pd.Series:
    """
    Very simple wind → power proxy:
    - cubic ramp up to rated speed (vr ~ 12 m/s),
    - cap at 1.0,
    - cut out above ~25 m/s.
    Returns a normalized proxy [0,1].
    """
    v = wind_ms.astype("float64").clip(lower=0)
    vr, vcut = 12.0, 25.0
    p = (v / vr) ** 3
    p = p.clip(upper=1.0)
    p = p.where(v < vcut, 0.0)
    p.name = "wind_proxy"
    return p
