#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.features.build_features import build_feature_table
from src.features.targets import make_day_ahead_target

RAW_PQ = Path("data/raw/semopx_dam_60min_hrp.parquet")
RAW_CSV = Path("data/raw/semopx_dam_60min_hrp.csv")
OUT_PQ = Path("data/processed/train.parquet")

IE_TZ = "Europe/Dublin"


def load_hrp60_file() -> pd.DataFrame:
    if RAW_PQ.exists():
        df = pd.read_parquet(RAW_PQ)
    elif RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
    else:
        raise SystemExit(
            "Missing SEMOpx HRP60 file.\n"
            "Run: python scripts/fetch_semopx_dam_60min_hrp.py --start YYYY-MM-DD --end YYYY-MM-DD\n"
            "Expected: data/raw/semopx_dam_60min_hrp.parquet (or .csv)"
        )

    # Normalize columns
    if "dam_60min_hrp_eur_mwh" in df.columns and "dam_eur_mwh" not in df.columns:
        df = df.rename(columns={"dam_60min_hrp_eur_mwh": "dam_eur_mwh"})

    if "ts_utc" not in df.columns or "dam_eur_mwh" not in df.columns:
        raise SystemExit(f"Bad HRP60 schema. Need columns ts_utc, dam_eur_mwh. Got: {list(df.columns)}")

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["dam_eur_mwh"] = pd.to_numeric(df["dam_eur_mwh"], errors="coerce")
    df = df.dropna(subset=["ts_utc", "dam_eur_mwh"]).sort_values("ts_utc")
    df = df.drop_duplicates("ts_utc", keep="last").reset_index(drop=True)
    return df


def main(days: int, horizon_hours: int) -> int:
    df = load_hrp60_file()

    # keep last N days
    cutoff = df["ts_utc"].max() - pd.Timedelta(days=days)
    df = df[df["ts_utc"] >= cutoff].copy()

    dam = (
        df.set_index("ts_utc")["dam_eur_mwh"]
        .tz_convert(IE_TZ)
        .tz_localize(None)
    )
    dam = dam[~dam.index.duplicated(keep="last")].sort_index()

    # No fundamentals in SEMOpx-only mode
    load_fc = pd.Series(index=dam.index, dtype=float, name="load_forecast_mw")
    ws_fc = pd.DataFrame(index=dam.index)
    weather = pd.DataFrame(index=dam.index)

    X = build_feature_table(dam, load_fc, ws_fc, weather)
    y = make_day_ahead_target(dam, horizon_hours=horizon_hours).reindex(X.index)

    out = X.copy()
    out["target"] = y
    out = out.dropna(subset=["target"])
    out = out.ffill().bfill().fillna(0)

    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PQ)
    print(f"✅ Wrote {len(out):,} rows -> {OUT_PQ}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365, help="How many days to keep in train.parquet")
    ap.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours")
    args = ap.parse_args()
    raise SystemExit(main(days=args.days, horizon_hours=args.horizon))
