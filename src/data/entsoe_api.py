# src/data/entsoe_api.py
from __future__ import annotations
import os
import pandas as pd
from zoneinfo import ZoneInfo
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# entsoe-py expects timestamps in Europe/Brussels (per docs/examples)
TZ_QUERY = ZoneInfo("Europe/Brussels")

class Entsoe:
    """Thin wrapper around entsoe-py with sane defaults for Ireland (IE/IE(SEM))."""

    def __init__(self, token: str | None = None, area: str = "IE(SEM)"):
        token = token or os.getenv("ENTSOE_TOKEN")
        if not token:
            raise RuntimeError("ENTSOE_TOKEN not set. Put it in .env or st.secrets.")
        self.client = EntsoePandasClient(api_key=token)
        # Prefer explicit IE(SEM). If your entsoe-py is older, 'IE' also works via mapping.
        self.area = area  # "IE(SEM)" or "IE"

    def _brussels(self, dt_like) -> pd.Timestamp:
        """Return tz-aware Timestamp in Europe/Brussels as required by entsoe-py."""
        ts = pd.Timestamp(dt_like)
        if ts.tzinfo is None:
            return ts.tz_localize(TZ_QUERY)
        return ts.tz_convert(TZ_QUERY)

    # ---------- Prices ----------
    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        s = self.client.query_day_ahead_prices(
            self.area,
            start=self._brussels(start),
            end=self._brussels(end),
        )
        if s is None or len(s) == 0:
            raise NoMatchingDataError("No day-ahead prices returned.")
        s.name = "dam_eur_mwh"
        return s

    # ---------- Load forecast ----------
    def load_forecast(self, start: str, end: str) -> pd.Series:
        s = self.client.query_load_forecast(
            self.area,
            start=self._brussels(start),
            end=self._brussels(end),
        )
        if s is None or len(s) == 0:
            raise NoMatchingDataError("No load forecast returned.")
        s.name = "load_forecast_mw"
        return s

    # ---------- Wind & Solar forecast (compose from generation forecast by PSR) ----------
    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
        - solar_mw
        - wind_onshore_mw
        - wind_offshore_mw
        - wind_total_mw
        """
        def _try(psr: str) -> pd.Series:
            try:
                s = self.client.query_generation_forecast(
                    self.area,
                    start=self._brussels(start),
                    end=self._brussels(end),
                    psr_type=psr,   # B16=Solar, B19=Wind Onshore, B18=Wind Offshore
                )
                return s if s is not None else pd.Series(dtype=float)
            except NoMatchingDataError:
                return pd.Series(dtype=float)

        solar = _try("B16")
        wind_on = _try("B19")
        wind_off = _try("B18")

        df = pd.concat(
            {
                "solar_mw": solar,
                "wind_onshore_mw": wind_on,
                "wind_offshore_mw": wind_off,
            },
            axis=1,
        ).sort_index()

        cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df]
        if cols:
            df["wind_total_mw"] = df[cols].sum(axis=1, min_count=1)
        return df
