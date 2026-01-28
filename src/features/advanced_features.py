from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.holidays_ie import is_ie_holiday


class AdvancedFeatureBuilder:
    """Irish electricity market-specific features."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_calendar_features(self) -> "AdvancedFeatureBuilder":
        idx = self.df.index

        self.df["hour"] = idx.hour
        self.df["dow"] = idx.dayofweek
        self.df["month"] = idx.month
        self.df["day_of_year"] = idx.dayofyear

        self.df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        self.df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        self.df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

        self.df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        self.df["is_holiday"] = idx.map(lambda x: int(is_ie_holiday(x.date())))
        self.df["is_dst"] = idx.map(
            lambda x: int(x.dst().total_seconds() > 0) if x.dst() else 0
        )

        self.df["is_morning_peak"] = ((idx.hour >= 7) & (idx.hour <= 9)).astype(int)
        self.df["is_evening_peak"] = ((idx.hour >= 17) & (idx.hour <= 21)).astype(int)
        self.df["is_night"] = ((idx.hour >= 23) | (idx.hour <= 5)).astype(int)

        return self

    def add_price_features(self, price_col: str = "dam_eur_mwh") -> "AdvancedFeatureBuilder":
        prices = self.df[price_col]

        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            self.df[f"price_lag_{lag}h"] = prices.shift(lag)

        self.df["price_same_hour_yesterday"] = prices.shift(24)
        self.df["price_same_hour_last_week"] = prices.shift(168)

        for window in [6, 12, 24, 48, 168]:
            self.df[f"price_roll_{window}h_mean"] = prices.rolling(window, min_periods=1).mean()
            self.df[f"price_roll_{window}h_std"] = prices.rolling(window, min_periods=1).std()
            self.df[f"price_roll_{window}h_min"] = prices.rolling(window, min_periods=1).min()
            self.df[f"price_roll_{window}h_max"] = prices.rolling(window, min_periods=1).max()

        self.df["price_momentum_24h"] = prices.diff(24)
        self.df["price_momentum_168h"] = prices.diff(168)

        self.df["price_volatility_24h"] = prices.rolling(24).std() / prices.rolling(24).mean()

        roll_min = prices.rolling(168, min_periods=24).min()
        roll_max = prices.rolling(168, min_periods=24).max()
        self.df["price_percentile_week"] = (prices - roll_min) / (roll_max - roll_min + 1e-6)

        return self

    def add_supply_features(self) -> "AdvancedFeatureBuilder":
        load = self.df["load_forecast_mw"] if "load_forecast_mw" in self.df.columns else None

        if "wind_total_mw" in self.df.columns and load is not None:
            wind = self.df["wind_total_mw"]
            load_safe = load.clip(lower=100)

            self.df["wind_penetration"] = wind / load_safe
            self.df["wind_penetration_lag24"] = self.df["wind_penetration"].shift(24)

            self.df["wind_change_6h"] = wind.diff(6)
            self.df["wind_change_24h"] = wind.diff(24)

            self.df["residual_load"] = load_safe - wind
            self.df["residual_load_lag24"] = self.df["residual_load"].shift(24)

        if "solar_mw" in self.df.columns and load is not None:
            load_safe = load.clip(lower=100)
            self.df["solar_penetration"] = self.df["solar_mw"] / load_safe

        return self

    def add_demand_features(self) -> "AdvancedFeatureBuilder":
        if "load_forecast_mw" not in self.df.columns:
            return self

        load = self.df["load_forecast_mw"]

        for lag in [1, 24, 48, 168]:
            self.df[f"load_lag_{lag}h"] = load.shift(lag)

        self.df["load_roll_24h_mean"] = load.rolling(24, min_periods=1).mean()
        self.df["load_roll_168h_mean"] = load.rolling(168, min_periods=24).mean()

        self.df["load_gradient_3h"] = load.diff(3)
        self.df["load_gradient_6h"] = load.diff(6)

        self.df["load_deviation_from_weekly"] = load - self.df["load_roll_168h_mean"]

        return self

    def add_weather_features(self) -> "AdvancedFeatureBuilder":
        if "temperature_2m" in self.df.columns:
            temp = self.df["temperature_2m"]
            self.df["hdd"] = np.maximum(15.5 - temp, 0)
            self.df["cdd"] = np.maximum(temp - 22, 0)
            self.df["temp_gradient_6h"] = temp.diff(6)
            self.df["temp_lag_24h"] = temp.shift(24)

        if "cloud_cover" in self.df.columns:
            self.df["clear_sky_index"] = 1 - self.df["cloud_cover"] / 100

        return self

    def add_interconnector_features(
        self,
        ewic_flow: pd.Series | None = None,
        moyle_flow: pd.Series | None = None,
    ) -> "AdvancedFeatureBuilder":
        if ewic_flow is not None:
            self.df["ewic_flow"] = ewic_flow.reindex(self.df.index)
            self.df["ewic_flow_lag24"] = self.df["ewic_flow"].shift(24)
            self.df["ie_net_importer"] = (self.df["ewic_flow"] > 0).astype(int)

        if moyle_flow is not None:
            self.df["moyle_flow"] = moyle_flow.reindex(self.df.index)

        return self

    def add_market_features(self) -> "AdvancedFeatureBuilder":
        idx = self.df.index
        self.df["hours_until_delivery"] = 24
        self.df["is_first_hour_of_day"] = (idx.hour == 0).astype(int)
        self.df["is_last_hour_of_day"] = (idx.hour == 23).astype(int)

        return self

    def build(self) -> pd.DataFrame:
        return self.df


def build_all_features(
    dam: pd.Series,
    load_fc: pd.Series,
    windsol_fc: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame(index=dam.index)
    df["dam_eur_mwh"] = dam
    df["load_forecast_mw"] = load_fc.reindex(df.index)

    for col in windsol_fc.columns:
        df[col] = windsol_fc[col].reindex(df.index)

    for col in weather.columns:
        df[col] = weather[col].reindex(df.index)

    builder = AdvancedFeatureBuilder(df)
    features = (
        builder.add_calendar_features()
        .add_price_features()
        .add_supply_features()
        .add_demand_features()
        .add_weather_features()
        .add_market_features()
        .build()
    )

    return features
