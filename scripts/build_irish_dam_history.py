from pathlib import Path
import pandas as pd

# project root = one level up from this script
BASE = Path(__file__).resolve().parents[1]

LB2 = BASE / "data" / "raw" / "Lookback2_mkt.xlsx"
LB1 = BASE / "data" / "raw" / "lookback_mkt.xlsx"
OUT = BASE / "data" / "raw" / "irish_dam_history.parquet"

# keep only recent history (you can relax this if you want)
START_DATE = "2018-10-01"   # SEMOpx go-live


def load_book(path: Path) -> pd.DataFrame:
    """
    Read one SEMOpx lookback workbook and return a DataFrame:

        index: ts_utc (datetime64[ns, UTC])
        column: dam_eur_mwh
    """
    print(f"Reading {path.name} ...")
    # sheet with auction results
    df = pd.read_excel(path, sheet_name="auctions_to", engine="openpyxl")

    # keep only DAM rows
    df = df[df["auction"].astype(str).str.upper().str.startswith("DAM")].copy()

    # timestamp column is already UTC in ISO format, e.g. 2021-01-01T00:00:00Z
    df["ts_utc"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # price column in EUR/MWh
    df["dam_eur_mwh"] = pd.to_numeric(df["price_eur"], errors="coerce")

    df = df[["ts_utc", "dam_eur_mwh"]].dropna(subset=["ts_utc", "dam_eur_mwh"])
    df = df.sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
    df = df.set_index("ts_utc")

    # make sure index is tz-aware UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    return df


def main():
    frames = []

    if LB1.exists():
        frames.append(load_book(LB1))
    if LB2.exists():
        frames.append(load_book(LB2))

    if not frames:
        raise SystemExit("No Lookback workbooks found in data/raw/. "
                         "Put Lookback2_mkt.xlsx and lookback_mkt.xlsx there.")

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # optional: drop very old history
    df = df[df.index >= pd.Timestamp(START_DATE, tz="UTC")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"✅ Saved {len(df):,} rows to {OUT}")


if __name__ == "__main__":
    main()
