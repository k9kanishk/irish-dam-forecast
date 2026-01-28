from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.base import DataQualityReport, DataSource
from src.data.entsoe_api import fetch_ie_dam_prices_entsoe
from src.data.validators import PriceDataValidator, quality_report_from_validator


class EntsoeDayAheadPriceSource(DataSource):
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path("data/processed")

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        return fetch_ie_dam_prices_entsoe(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            cache_dir=self.cache_dir,
        )

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        validator = PriceDataValidator(df)
        return quality_report_from_validator(validator)
