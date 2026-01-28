from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.base import DataQualityReport, DataSource
from src.data.eirgrid_prices import fetch_dam_recent
from src.data.validators import PriceDataValidator, quality_report_from_validator


class EirGridDayAheadPriceSource(DataSource):
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        days = max(1, (end - start).days + 1)
        df = fetch_dam_recent(days=days)
        df = df.sort_index()
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask]

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        validator = PriceDataValidator(df)
        return quality_report_from_validator(validator)
