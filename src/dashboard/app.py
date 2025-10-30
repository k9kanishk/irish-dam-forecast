# src/dashboard/app.py

# --- bootstrap (imports first!) ---
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import yaml
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is on sys.path so `import src...` works in Streamlit Cloud/Codespaces
APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[2]  # .../irish-dam-forecast
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

load_dotenv()  # local dev uses .env; on Streamlit Cloud we'll use st.secrets

DATA_PATH = Path("data/processed/train.parquet")

from datetime import timedelta
from entsoe.exceptions import NoMatchingDataError

def _read_cached_dam(start: str, end: str) -> pd.Series:
    """Get dam_eur_mwh from our parquet if API is missing."""
    if not DATA_PATH.exists():
        return pd.Series(dtype=float)
    dfc = pd.read_parquet(DATA_PATH)
    if "dam_eur_mwh" not in dfc.columns:
        return pd.Series(dtype=float)
    s = dfc["dam_eur_mwh"].copy()
    s.index = pd.to_datetime(s.index)
    s = s[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]
    s.name = "dam_eur_mwh"
    return s

def _get_dam_history(ent, start: str, end: str) -> pd.Series:
    """Try API, then step the end date back, then fall back to parquet cache."""
    # 1) try the full window
    try:
        s = ent.day_ahead_prices(start=start, end=end)
        if s is not None and len(s):
            return s
    except NoMatchingDataError:
        pass
    # 2) back off the end date up to 14 days
    for back in range(1, 15):
        try:
            end2 = str((pd.to_datetime(end) - pd.Timedelta(days=back)).date())
            if pd.to_datetime(end2) < pd.to_datetime(start):
                break
            s = ent.day_ahead_prices(start=start, end=end2)
            if s is not None and len(s):
                return s
        except NoMatchingDataError:
            continue
    # 3) fallback to cached parquet (best-effort)
    s = _read_cached_dam(start, end)
    return s


# put near the top of app.py, after DATA_PATH is defined
with st.sidebar:
    if st.button("Rebuild data"):
        DATA_PATH.unlink(missing_ok=True)
        st.cache_data.clear()
        st.rerun()



def _as_date(s):
    """Parse s to date; return None if missing/blank/invalid."""
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _chunk_edges(start_date, end_date, days=30):  # 30 keeps calls small/fast
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    step = pd.Timedelta(days=days)
    cur = pd.to_datetime(s); last = pd.to_datetime(e)
    while cur <= last:
        nxt = min(cur + step, last)
        yield str(cur.date()), str(nxt.date())
        cur = nxt + pd.Timedelta(days=1)


@st.cache_data(show_spinner=True)
def ensure_dataset():
    """Build data/processed/train.parquet once, using ENTSO-E + Open-Meteo, chunked and tolerant to empty chunks."""
    if DATA_PATH.exists():
        return  # already built

    # Heavy imports only when needed (after sys.path fix above)
    from src.data.entsoe_api import Entsoe
    from src.data.weather import fetch_hourly
    from src.features.build_features import build_feature_table
    from src.features.targets import make_day_ahead_target
    from entsoe.exceptions import NoMatchingDataError

    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Compute window (defaults if missing)
    start_cfg = _as_date(cfg.get("train", {}).get("start"))
    end_cfg   = _as_date(cfg.get("train", {}).get("end"))
    if not end_cfg:
        end_cfg = datetime.now(timezone.utc).date()
    if not start_cfg:
        start_cfg = (pd.to_datetime(end_cfg) - pd.Timedelta(days=90)).date()
    start_all = str(start_cfg)
    end_all   = str(end_cfg)

    # --- FAST MODE: clamp window to last N days so first build is quick ---
    MAX_DAYS = int(os.getenv("MAX_DAYS", "180"))  # tune 30–90 as you like
    span_days = (pd.to_datetime(end_all) - pd.to_datetime(start_all)).days + 1
    if span_days > MAX_DAYS:
        start_all = str((pd.to_datetime(end_all) - pd.Timedelta(days=MAX_DAYS)).date())
        st.info(f"Fast mode: clamped window to last {MAX_DAYS} days → {start_all} → {end_all}")

    # Token: ENV first, then Streamlit secrets
    token = os.getenv("ENTSOE_TOKEN") or st.secrets.get("ENTSOE_TOKEN")
    if not token:
        st.error("ENTSOE_TOKEN not found. Set it in .env (local) or in Streamlit → Settings → Secrets.")
        st.stop()

    ent = Entsoe(token=token)

    # ---- Chunked pulls to avoid 400s and tolerate empty chunks ----
    def _pull_series(method_name: str) -> pd.Series:
        parts, empty_spans = [], []
        for i, (s, e) in enumerate(_chunk_edges(start_all, end_all, days=30), 1):
            # Optional: comment out next line if you don’t want logs on the page
            st.write(f"ENTSO-E {method_name} chunk {i}: {s} → {e}")
            for attempt in range(3):
                try:
                    srs = getattr(ent, method_name)(start=s, end=e)
                    if srs is not None and len(srs) > 0:
                        parts.append(srs)
                    else:
                        empty_spans.append((s, e))
                    break
                except NoMatchingDataError:
                    empty_spans.append((s, e))
                    break
                except Exception:
                    time.sleep(1.5 * (attempt + 1))
                    if attempt == 2:
                        raise
        if not parts:
            return pd.Series(dtype=float)
        srs = pd.concat(parts).sort_index()
        srs = srs[~srs.index.duplicated(keep="last")]
        return srs

    def _pull_frame(method_name: str) -> pd.DataFrame:
        parts, empty_spans = [], []
        for i, (s, e) in enumerate(_chunk_edges(start_all, end_all, days=30), 1):
            # Optional: comment out next line if you don’t want logs on the page
            st.write(f"ENTSO-E {method_name} chunk {i}: {s} → {e}")
            for attempt in range(3):
                try:
                    dfp = getattr(ent, method_name)(start=s, end=e)
                    if dfp is not None and not dfp.empty:
                        parts.append(dfp)
                    else:
                        empty_spans.append((s, e))
                    break
                except NoMatchingDataError:
                    empty_spans.append((s, e))
                    break
                except Exception:
                    time.sleep(1.5 * (attempt + 1))
                    if attempt == 2:
                        raise
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    st.caption(f"Fetching ENTSO-E (chunked): {start_all} → {end_all}")
    dam     = _pull_series("day_ahead_prices")
    load_fc = _pull_series("load_forecast")
    ws      = _pull_frame("wind_solar_forecast")

    # Hard guards
    if dam.empty:
        st.error("No day-ahead prices returned. Set a recent window in config.yaml (e.g., last 90–180 days) and rerun.")
        st.stop()
    if load_fc.empty:
        st.error("No load forecast returned. Try a recent window in config.yaml and rerun.")
        st.stop()

    # Weather
    lat = cfg["weather"]["lat"]; lon = cfg["weather"]["lon"]
    weather = fetch_hourly(lat, lon, start_all, end_all)

    # ---- Timezone harmonization (make all indices Europe/Dublin tz-naive) ----
    def _to_local_naive(obj, assume_utc: bool = False):
        if not hasattr(obj, "index") or not isinstance(obj.index, pd.DatetimeIndex):
            return obj
        out = obj.copy()
        idx = out.index
        if idx.tz is None:
            base_tz = "UTC" if assume_utc else "Europe/Brussels"
            idx = idx.tz_localize(base_tz)
        idx = idx.tz_convert("Europe/Dublin").tz_localize(None)
        out.index = idx
        return out

    dam     = _to_local_naive(dam, assume_utc=False)
    load_fc = _to_local_naive(load_fc, assume_utc=False)
    ws      = _to_local_naive(ws, assume_utc=False)
    weather = _to_local_naive(weather, assume_utc=True)

    # Overlay when actuals exist
    if len(X_day):
        preds = mdl.predict(X_day)
        out = pd.DataFrame({"ts": X_day.index, "forecast_eur_mwh": preds}).set_index("ts")
        st.subheader("Hourly forecast")
        st.dataframe(out)
        
        # if the selected date <= last available DAM date, overlay actuals
        last_actual = df.index.max()
        if out.index.max() <= last_actual:
            # 'dam' series already fetched when building the dataset; reload from parquet
            hist = pd.read_parquet(DATA_PATH)
            actual = hist.reindex(out.index)["target"]  # target was next-day price
            both = out.join(actual.rename("actual_eur_mwh"))
            st.plotly_chart(px.line(both, y=["forecast_eur_mwh","actual_eur_mwh"]),use_container_width=True)
        else:
            st.info("No official DAM price yet for this day – showing forecast only.")


        
    
    # ---- Ensure load_fc is a Series named 'load_forecast_mw' ----
    if isinstance(load_fc, pd.DataFrame):
        def _norm(c): return " ".join(map(str, c)).lower() if isinstance(c, tuple) else str(c).lower()
        colmap = {_norm(c): c for c in load_fc.columns}
        pick = None
        for key in ("load forecast", "forecast", "load"):
            for k, c in colmap.items():
                if key in k:
                    pick = c; break
            if pick is not None:
                break
        if pick is None:
            pick = load_fc.columns[0]
        load_fc = load_fc[pick]
    load_fc = pd.to_numeric(load_fc, errors="coerce").astype("float64")
    load_fc.name = "load_forecast_mw"

    # ---- Build features + target ----
        # ---- Build features + target ----
    # Align all to the common intersection first (prevents widespread NaNs)
    common_idx = dam.index.intersection(load_fc.index)
    if not ws.empty:
        common_idx = common_idx.intersection(ws.index)
    common_idx = common_idx.intersection(weather.index)

    dam     = dam.reindex(common_idx)
    load_fc = load_fc.reindex(common_idx)
    ws      = ws.reindex(common_idx) if not ws.empty else ws
    weather = weather.reindex(common_idx)

    X = build_feature_table(dam, load_fc, ws, weather)
    y = make_day_ahead_target(dam).reindex(X.index)

    # Keep only rows that must exist for training/forecasting
    must_have = ["dam_eur_mwh", "load_forecast_mw"]
    keep = y.notna()
    for c in must_have:
        if c in X.columns:
            keep &= X[c].notna()

    X = X.loc[keep].copy()
    y = y.loc[keep]

    # For the rest of feature columns, fill gaps (don’t nuke the dataset)
    filler_cols = [c for c in X.columns if c not in must_have]
    if filler_cols:
        X[filler_cols] = X[filler_cols].interpolate(limit_direction="both")
        X[filler_cols] = X[filler_cols].fillna(method="ffill").fillna(method="bfill")

    if X.empty:
        st.error(
            "No overlapping timestamps across ENTSO-E & weather feeds for the chosen window. "
            "Pick a recent window in config.yaml (e.g., 2025-07-01 → 2025-09-30) and rerun."
        )
        st.stop()

    out = X.copy()
    out["target"] = y

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_PATH)  # NOTE: no global dropna() anymore



# Build (cached) then load
ensure_dataset()
df = pd.read_parquet(DATA_PATH)

# --- app proper (keep a single date picker) ---
import plotly.express as px
from datetime import date as _date
from zoneinfo import ZoneInfo

# compute coverage from the DataFrame index
idx = pd.DatetimeIndex(df.index)
days = pd.Index(idx.date)
if len(days) == 0:
    st.error("No rows found in the dataset.")
    st.stop()

def _as_pydate(d):
    return d if isinstance(d, _date) else pd.to_datetime(d).date()

day_min = _as_pydate(days.min())
day_max = _as_pydate(days.max())
today_ie = datetime.now(ZoneInfo("Europe/Dublin")).date()
default_day = max(day_min, min(day_max, today_ie))

st.caption(f"Data coverage: {day_min} → {day_max} | rows: {len(df):,}")

date = st.date_input(
    "Choose a date to forecast",
    value=default_day,
    min_value=day_min,
    max_value=day_max,
)

# Train quick model on all rows
y = df.pop("target")
mdl = __import__("src.models.xgb_model", fromlist=["make_model"]).make_model()
mdl.fit(df, y)

st.title("Irish Day-Ahead Power Price Forecast (SEMOpx)")

X_day = df.loc[df.index.date == pd.to_datetime(date).date()]
if len(X_day):
    preds = mdl.predict(X_day)
    out = pd.DataFrame({"ts": X_day.index, "forecast_eur_mwh": preds}).set_index("ts")
    st.subheader("Hourly forecast")
    st.dataframe(out)
    st.plotly_chart(px.line(out, y="forecast_eur_mwh"), use_container_width=True)
else:
    st.info("No features for this day yet. Adjust the training window in config.yaml or fetch more data.")


# --- Forecast mode: predict for a future calendar day (e.g., tomorrow) ---
from datetime import date as _date, timedelta
from src.data.entsoe_api import Entsoe
from src.data.weather import fetch_hourly
from src.features.build_features import build_feature_table

def make_features_for_day(target_day: _date) -> pd.DataFrame:
    """Build X for target_day using: recent DAM (for lags),
    ENTSO-E forecasts on target_day, and weather for target_day."""
    token = os.getenv("ENTSOE_TOKEN") or st.secrets.get("ENTSOE_TOKEN")
    ent = Entsoe(token=token)

    # recent history for DAM lags
    start_hist = (pd.to_datetime(target_day) - pd.Timedelta(days=14)).date()
    end_hist   = (pd.to_datetime(target_day) - pd.Timedelta(days=1)).date()
    dam_hist   = _get_dam_history(ent, str(start_hist), str(end_hist))

    # If we truly have no DAM at all, we can still try but warn that lags will be NaN
    if dam_hist.empty:
        st.warning("DAM history missing for recent days; continuing without price lags.")

    # exogenous forecasts ON the target day
    try:
        load_fc = ent.load_forecast(start=str(target_day), end=str(target_day))
    except NoMatchingDataError:
        st.warning("ENTSO-E load forecast not available for the selected day yet.")
        return pd.DataFrame()

    try:
        ws_fc = ent.wind_solar_forecast(start=str(target_day), end=str(target_day))
    except NoMatchingDataError:
        st.warning("ENTSO-E wind/solar forecast not available for the selected day yet.")
        return pd.DataFrame()

    # weather (Open-Meteo) for the target window
    # (you can read lat/lon from config.yaml; hard-coded here for brevity)
    lat, lon = 53.4, -8.2
    weather = fetch_hourly(lat, lon, str(target_day), str(target_day))

    # --- timezone harmonisation like training ---
    def _tz_fix(obj, assume_utc=False):
        if not hasattr(obj, "index") or not isinstance(obj.index, pd.DatetimeIndex):
            return obj
        idx = obj.index
        if idx.tz is None:
            idx = idx.tz_localize("Europe/Brussels" if not assume_utc else "UTC")
        idx = idx.tz_convert("Europe/Dublin").tz_localize(None)
        out = obj.copy(); out.index = idx
        return out

    dam_hist = _tz_fix(dam_hist)
    load_fc  = _tz_fix(load_fc)
    ws_fc    = _tz_fix(ws_fc)
    weather  = _tz_fix(weather, assume_utc=True)

    # Ensure a Series named load_forecast_mw
    if isinstance(load_fc, pd.DataFrame):
        # heuristics: take the first numeric column
        load_fc = pd.to_numeric(load_fc.iloc[:, 0], errors="coerce")
    load_fc.name = "load_forecast_mw"

    # Build features (same function as training)
    X_full = build_feature_table(dam_hist, load_fc, ws_fc, weather)

    # Keep only the target day’s rows
    mask = X_full.index.date == target_day
    return X_full.loc[mask]


# --- UI: Tomorrow button
with st.sidebar:
    if st.button("Forecast tomorrow (T+1)"):
        tgt = (datetime.now(ZoneInfo("Europe/Dublin")).date() + timedelta(days=1))
        X_t1 = make_features_for_day(tgt)
        if len(X_t1):
            preds = mdl.predict(X_t1)
            out_f = pd.DataFrame({"ts": X_t1.index, "forecast_eur_mwh": preds}).set_index("ts")
            st.success(f"Forecast for {tgt}:")
            st.dataframe(out_f)
            st.plotly_chart(px.line(out_f, y="forecast_eur_mwh"), use_container_width=True)
        else:
            st.warning("Could not build features for tomorrow yet (inputs not available). Try later.")


# --- Backtest (last N days) ---
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def backtest_last_n_days(df_full: pd.DataFrame, n_days: int = 14):
    data = df_full.copy()
    y = data.pop("target")
    X = data
    cutoff = (X.index.max().date() - pd.Timedelta(days=n_days)).date()
    train = X.index.date < cutoff
    Xtr, ytr, Xte, yte = X.loc[train], y.loc[train], X.loc[~train], y.loc[~train]
    mdl_bt = __import__("src.models.xgb_model", fromlist=["make_model"]).make_model()
    mdl_bt.fit(Xtr, ytr)
    pred = mdl_bt.predict(Xte)
    rmse = mean_squared_error(yte, pred, squared=False)
    mape = np.mean(np.abs((yte - pred) / np.clip(np.abs(yte), 1e-6, None))) * 100
    return rmse, mape, pd.DataFrame({"actual": yte, "pred": pred}, index=Xte.index).sort_index()

with st.sidebar:
    if st.button("Run 14-day backtest"):
        rmse, mape, df_eval = backtest_last_n_days(df)
        st.success(f"RMSE: {rmse:,.2f}  |  MAPE: {mape:,.2f}%")
        st.line_chart(df_eval.resample("H").mean())


