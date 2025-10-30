# src/data/entsoe_api.py
from __future__ import annotations
import os
import pandas as pd
from zoneinfo import ZoneInfo
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

AREA_IE = "IE_SEM"                     # <- use IE (works across entsoe-py versions)
TZ_QUERY = ZoneInfo("Europe/Brussels")  # entsoe-py expects Brussels tz

class Entsoe:
    def _try_pairs(self, func, pairs, **kwargs):
        """Try several border code pairs until one works; return first Series/DataFrame."""
        for a, b in pairs:
            try:
                return func(a, b, **kwargs)
            except NoMatchingDataError:
                continue
            except Exception:
                continue
        # nothing worked
        raise NoMatchingDataError

    def net_imports(self, start: str, end: str) -> pd.Series:
        """
        Net imports into IE_SEM from GB. Positive = importing into IE.
        """
        # pairs to try (ENTSO-E codes vary by border)
        pairs = [
            ("GB", "IE_SEM"),
            ("GB_GBN", "IE_SEM"),
            ("GB_NIR", "IE_SEM"),
        ]

        # flow INTO IE (GB -> IE)
        s_in = self._try_pairs(
            self.client.query_crossborder_flows,
            pairs=[(a, b) for a, b in pairs],
            start=self._brussels(start),
            end=self._brussels(end),
        )
        # flow OUT OF IE (IE -> GB)
        s_out = self._try_pairs(
            self.client.query_crossborder_flows,
            pairs=[(b, a) for a, b in pairs],
            start=self._brussels(start),
            end=self._brussels(end),
        )
        s = s_in.rename("flow_in") - s_out.rename("flow_out")
        s.name = "net_imports_mw"
        return s
    def __init__(self, token: str | None = None, area: str = AREA_IE):
        token = token or os.getenv("ENTSOE_TOKEN")
        if not token:
            raise RuntimeError("ENTSOE_TOKEN not set. Put it in .env or st.secrets.")
        self.client = EntsoePandasClient(api_key=token, timeout=30, retry_count=1)
        self.area = area

    def _brussels(self, dt_like) -> pd.Timestamp:
        ts = pd.Timestamp(dt_like)
        if ts.tzinfo is None:
            return ts.tz_localize(TZ_QUERY)
        return ts.tz_convert(TZ_QUERY)

    # -------- Prices --------
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

    # --- Load forecast -> always return a Series named 'load_forecast_mw' ---
    def load_forecast(self, start: str, end: str) -> pd.Series:
        df = self.client.query_load_and_forecast(self.area,start=self._brussels(start), end=self._brussels(end),)

        def _norm(c): return " ".join(map(str, c)).lower()
        cand = [c for c in df.columns if "forecast" in _norm(c)]
        col = cand[0] if cand else df.columns[-1]
        s = pd.to_numeric(df[col], errors="coerce")
        s.name = "load_forecast_mw"
        return s
    
    

    
   



    # -------- Wind & Solar forecast (compose from PSR types) --------
    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        """
        Columns:
          - solar_mw
          - wind_onshore_mw
          - wind_offshore_mw
          - wind_total_mw
        """
        def _try(psr: str) -> pd.Series:
            try:
                s = self.client.query_wind_and_solar_forecast(
                    self.area,
                    start=self._brussels(start),
                    end=self._brussels(end),
                    psr_type=psr,    # B16=Solar, B19=Wind Onshore, B18=Wind Offshore
                )
                return s if s is not None else pd.Series(dtype=float)
            except NoMatchingDataError:
                return pd.Series(dtype=float)

        solar = _try("B16")
        wind_on = _try("B19")
        wind_off = _try("B18")

        df = pd.concat(
            {"solar_mw": solar, "wind_onshore_mw": wind_on, "wind_offshore_mw": wind_off},
            axis=1,
        ).sort_index()

        cols = [c for c in ["wind_onshore_mw", "wind_offshore_mw"] if c in df]
        if cols:
            df["wind_total_mw"] = df[cols].sum(axis=1, min_count=1)
        return df
