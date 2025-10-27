# src/data/entsoe_api.py
from __future__ import annotations
import os
import pandas as pd
from entsoe import EntsoePandasClient
from zoneinfo import ZoneInfo
from entsoe.exceptions import NoMatchingDataError

IE = "IE"  # Ireland bidding zone / country code


class Entsoe:
    def __init__(self, token: str | None = None, tz_local: str = "Europe/Dublin"):
        token = token or os.getenv("ENTSOE_TOKEN")
        if not token:
            raise RuntimeError("ENTSOE_TOKEN not set. Put it in .env or st.secrets.")
        self.client = EntsoePandasClient(api_key=token)
        self.tz_local = ZoneInfo(tz_local)
        self.tz_utc = ZoneInfo("UTC")

    def _utc(self, dt_like) -> pd.Timestamp:
        """Convert any datetime-like to tz-aware UTC Timestamp as required by entsoe-py."""
        ts = pd.Timestamp(dt_like)
        if ts.tzinfo is None:
            ts = ts.tz_localize(self.tz_local).tz_convert(self.tz_utc)
        else:
            ts = ts.tz_convert(self.tz_utc)
        return ts

    # ---------- Prices ----------
    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        s = self.client.query_day_ahead_prices(
            IE,
            start=self._utc(start),
            end=self._utc(end),
        )
        s.name = "dam_eur_mwh"
        return s

    # ---------- Load forecast ----------
    def load_forecast(self, start: str, end: str) -> pd.Series:
        s = self.client.query_load_forecast(
            IE,
            start=self._utc(start),
            end=self._utc(end),
        )
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
                    IE,
                    start=self._utc(start),
                    end=self._utc(end),
                    psr_type=psr,   # B16=Solar, B19=Wind Onshore, B18=Wind Offshore
                )
                # entsoe-py may return None/empty on gaps
                if s is None:
                    return pd.Series(dtype=float)
                return s
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

        # Sum on/offshore when available
        if "wind_onshore_mw" in df or "wind_offshore_mw" in df:
            cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df]
            if cols:
                df["wind_total_mw"] = df[cols].sum(axis=1, min_count=1)
        return df
