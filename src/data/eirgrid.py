# src/data/eirgrid.py
import pandas as pd
from pathlib import Path

def load_eirgrid_folder(folder="data/eirgrid") -> pd.DataFrame:
    """
    Read all CSVs you export from EirGrid portal into a tidy hourly dataframe.
    Expected columns include date/time + demand (MW) + wind (MW) + solar (MW) if available.
    We align to Europe/Dublin and return a DatetimeIndex.
    """
    p = Path(folder)
    if not p.exists():
        return pd.DataFrame()

    parts = []
    for f in p.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            # normalize common column names
            def _safelower(c):
                try:
                    return str(c).lower().strip()
                except Exception:
                    return ""  # fallback for weird labels
                    
            cols = {_safelower(c): c for c in df.columns}

            # try to locate time/demand/wind/solar
            tcol = next(c for k,c in cols.items() if "time" in k or "datetime" in k or "date" in k)
            df["ts"] = pd.to_datetime(df[tcol])
            for key, outname in [("demand","eirgrid_demand_mw"),
                                 ("load","eirgrid_demand_mw"),
                                 ("wind","eirgrid_wind_mw"),
                                 ("solar","eirgrid_solar_mw")]:
                for k,c in cols.items():
                    if key in k and c != tcol:
                        df[outname] = pd.to_numeric(df[c], errors="coerce")
            parts.append(df[["ts"]+[c for c in df.columns if c.endswith("_mw")]])
        except Exception:
            continue

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts).sort_values("ts")
    out = out.set_index(pd.DatetimeIndex(out["ts"])).drop(columns=["ts"])
    # ensure Europe/Dublin naive
    if out.index.tz is None:
        out.index = out.index.tz_localize("Europe/Dublin").tz_convert("Europe/Dublin").tz_localize(None)
    else:
        out.index = out.index.tz_convert("Europe/Dublin").tz_localize(None)
    # de-duplicate
    out = out[~out.index.duplicated(keep="last")]
    return out
