from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd

from .base import DataQualityReport, DataSource
from .cache import DataCache


@dataclass
class PipelineResult:
    data: pd.DataFrame
    quality: DataQualityReport
    cache_hit: bool


class DataPipeline:
    """Orchestrate fetch, validate, and cache for a single data source."""

    def __init__(
        self,
        source: DataSource,
        cache: Optional[DataCache] = None,
        cache_key: str = "data",
    ) -> None:
        self.source = source
        self.cache = cache
        self.cache_key = cache_key

    def run(
        self,
        start: datetime,
        end: datetime,
        fallback_sources: Iterable[DataSource] = (),
    ) -> PipelineResult:
        if self.cache is not None:
            cached = self.cache.get(self.cache_key)
            if cached.hit and cached.data is not None:
                quality = self.source.validate(cached.data)
                return PipelineResult(data=cached.data, quality=quality, cache_hit=True)

        df = self.source.fetch_with_fallback(start, end, fallback_sources)
        quality = self.source.validate(df)

        if self.cache is not None:
            self.cache.set(self.cache_key, df)

        return PipelineResult(data=df, quality=quality, cache_hit=False)
