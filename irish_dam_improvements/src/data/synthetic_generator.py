# src/data/synthetic_generator.py
"""
Realistic Synthetic Data Generator for Irish DAM Prices.

This generates synthetic but realistic electricity price data based on:
- Irish market characteristics (high wind penetration, ~40%)
- Seasonal patterns
- Intraday patterns (morning/evening peaks)
- Weekend effects
- Weather correlations

This is useful for:
1. Demo/portfolio purposes when real data isn't available
2. Testing the forecasting pipeline
3. Developing new features

The synthetic data mimics real Irish DAM price patterns observed in 2023-2025.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import hashlib


# Irish market price characteristics (based on 2023-2025 data)
PRICE_PARAMS = {
    "base_price": 100.0,         # Base price €/MWh
    "seasonal_amplitude": 30.0,  # Winter vs summer variation
    "daily_amplitude": 25.0,     # Intraday variation
    "wind_effect": -40.0,        # Price drop per unit wind penetration
    "noise_std": 15.0,           # Random noise
    "spike_prob": 0.02,          # Probability of price spike
    "spike_magnitude": 100.0,    # Additional price during spike
    "negative_prob": 0.01,       # Probability of negative prices (high wind)
    "weekend_discount": 0.85,    # Weekend prices ~85% of weekday
}

# Load pattern (MW) - typical Irish demand
LOAD_PARAMS = {
    "base_load": 3500.0,         # Base load MW
    "seasonal_amplitude": 1000.0, # Winter higher than summer
    "daily_amplitude": 1500.0,    # Intraday variation
    "noise_std": 200.0,
    "weekend_factor": 0.85,
}

# Wind pattern (MW)
WIND_PARAMS = {
    "mean_output": 1500.0,       # Mean wind output
    "std_output": 800.0,         # High variability
    "seasonal_factor_winter": 1.3,
    "seasonal_factor_summer": 0.7,
    "autocorrelation": 0.9,      # Wind is correlated hour-to-hour
}


def generate_hourly_index(
    start: str | datetime = None,
    end: str | datetime = None,
    days: int = 90
) -> pd.DatetimeIndex:
    """Generate hourly datetime index in UTC."""
    if end is None:
        end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        end = pd.Timestamp(end)
    
    if start is None:
        start = end - timedelta(days=days)
    else:
        start = pd.Timestamp(start)
    
    return pd.date_range(start=start, end=end, freq="H", tz="UTC")


def generate_wind(index: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    """
    Generate realistic wind generation time series.
    
    Irish wind has:
    - High variability
    - Seasonal pattern (more wind in winter)
    - Temporal autocorrelation
    """
    np.random.seed(seed)
    n = len(index)
    
    # Base wind with autocorrelation
    noise = np.random.randn(n)
    wind = np.zeros(n)
    wind[0] = WIND_PARAMS["mean_output"]
    
    for i in range(1, n):
        wind[i] = (
            WIND_PARAMS["autocorrelation"] * wind[i-1] +
            (1 - WIND_PARAMS["autocorrelation"]) * WIND_PARAMS["mean_output"] +
            WIND_PARAMS["std_output"] * (1 - WIND_PARAMS["autocorrelation"]**2)**0.5 * noise[i]
        )
    
    # Add seasonal pattern (more wind in winter)
    day_of_year = index.dayofyear
    seasonal = np.where(
        (index.month >= 11) | (index.month <= 2),
        WIND_PARAMS["seasonal_factor_winter"],
        np.where(
            (index.month >= 5) & (index.month <= 8),
            WIND_PARAMS["seasonal_factor_summer"],
            1.0
        )
    )
    wind *= seasonal
    
    # Clip to realistic range (0 to ~3500 MW installed capacity)
    wind = np.clip(wind, 0, 3500)
    
    return pd.Series(wind, index=index, name="wind_total_mw")


def generate_load(index: pd.DatetimeIndex, seed: int = 43) -> pd.Series:
    """
    Generate realistic load forecast time series.
    
    Irish load has:
    - Clear intraday pattern (peaks morning and evening)
    - Seasonal pattern (higher in winter)
    - Weekend reduction
    """
    np.random.seed(seed)
    n = len(index)
    
    # Hour of day effect
    hour = index.hour
    hour_effect = np.where(
        (hour >= 7) & (hour <= 9), 1.15,  # Morning peak
        np.where(
            (hour >= 17) & (hour <= 21), 1.2,  # Evening peak
            np.where(
                (hour >= 23) | (hour <= 5), 0.75,  # Night
                1.0
            )
        )
    )
    
    # Seasonal effect (winter higher)
    day_of_year = index.dayofyear
    seasonal = LOAD_PARAMS["seasonal_amplitude"] * np.cos(
        2 * np.pi * (day_of_year - 15) / 365  # Peak around Jan 15
    )
    
    # Weekend effect
    is_weekend = index.dayofweek >= 5
    weekend_factor = np.where(is_weekend, LOAD_PARAMS["weekend_factor"], 1.0)
    
    # Combine
    load = (
        LOAD_PARAMS["base_load"] + seasonal
    ) * hour_effect * weekend_factor
    
    # Add noise
    load += np.random.randn(n) * LOAD_PARAMS["noise_std"]
    
    # Clip to realistic range
    load = np.clip(load, 2000, 6000)
    
    return pd.Series(load, index=index, name="load_forecast_mw")


def generate_prices(
    index: pd.DatetimeIndex,
    wind: pd.Series,
    load: pd.Series,
    seed: int = 44
) -> pd.Series:
    """
    Generate realistic DAM prices based on fundamentals.
    
    Irish prices are driven by:
    - Gas prices (base)
    - Wind penetration (negative correlation)
    - Load level
    - Time of day
    - Weekend effects
    - Occasional spikes/negative prices
    """
    np.random.seed(seed)
    n = len(index)
    
    # Base price
    price = np.full(n, PRICE_PARAMS["base_price"])
    
    # Seasonal effect (higher in winter due to gas prices)
    day_of_year = index.dayofyear
    seasonal = PRICE_PARAMS["seasonal_amplitude"] * np.cos(
        2 * np.pi * (day_of_year - 15) / 365
    )
    price += seasonal
    
    # Intraday effect
    hour = index.hour
    daily = PRICE_PARAMS["daily_amplitude"] * np.sin(
        np.pi * (hour - 4) / 12  # Peak around 16:00
    )
    daily = np.where((hour >= 7) & (hour <= 21), daily, -10)  # Lower at night
    price += daily
    
    # Wind effect (higher wind = lower prices)
    wind_penetration = wind / load.clip(lower=2000)
    price += PRICE_PARAMS["wind_effect"] * wind_penetration
    
    # Weekend discount
    is_weekend = index.dayofweek >= 5
    price = np.where(is_weekend, price * PRICE_PARAMS["weekend_discount"], price)
    
    # Load effect
    load_factor = (load - LOAD_PARAMS["base_load"]) / 1000
    price += 10 * load_factor
    
    # Random noise
    price += np.random.randn(n) * PRICE_PARAMS["noise_std"]
    
    # Occasional spikes (e.g., plant outages, cold snaps)
    spikes = np.random.rand(n) < PRICE_PARAMS["spike_prob"]
    price = np.where(spikes, price + PRICE_PARAMS["spike_magnitude"], price)
    
    # Occasional negative prices (very high wind)
    high_wind = wind_penetration > 0.7
    neg_chance = np.random.rand(n) < PRICE_PARAMS["negative_prob"]
    price = np.where(high_wind & neg_chance, np.random.uniform(-20, 0, n), price)
    
    # Clip to market bounds (-500 to 4000 €/MWh)
    price = np.clip(price, -500, 4000)
    
    return pd.Series(price, index=index, name="dam_eur_mwh")


def generate_weather(index: pd.DatetimeIndex, seed: int = 45) -> pd.DataFrame:
    """Generate weather data (temperature, wind speed, cloud cover)."""
    np.random.seed(seed)
    n = len(index)
    
    # Temperature (°C) - seasonal + diurnal
    day_of_year = index.dayofyear
    hour = index.hour
    
    temp_seasonal = 10 + 8 * np.cos(2 * np.pi * (day_of_year - 200) / 365)  # Peak in July
    temp_diurnal = 3 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at noon
    temp = temp_seasonal + temp_diurnal + np.random.randn(n) * 2
    
    # Wind speed at 100m (m/s)
    wind_speed = 8 + 4 * np.random.randn(n)
    wind_speed = np.clip(wind_speed, 0, 25)
    
    # Cloud cover (%)
    cloud = 50 + 30 * np.random.randn(n)
    cloud = np.clip(cloud, 0, 100)
    
    return pd.DataFrame({
        "temperature_2m": temp,
        "wind100m_ms": wind_speed,
        "cloud_cover": cloud,
    }, index=index)


def generate_full_dataset(
    start: str | datetime = None,
    end: str | datetime = None,
    days: int = 90,
    seed: int = 42,
    save_to: Optional[Path] = None
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Generate a complete synthetic dataset for Irish DAM forecasting.
    
    Args:
        start: Start date (default: N days before end)
        end: End date (default: yesterday)
        days: Number of days if start not specified
        seed: Random seed for reproducibility
        save_to: Optional directory to save outputs
        
    Returns:
        Dict with keys: prices, load, wind, weather, features
    """
    # Generate index
    index = generate_hourly_index(start=start, end=end, days=days)
    
    # Generate components
    wind = generate_wind(index, seed=seed)
    load = generate_load(index, seed=seed + 1)
    prices = generate_prices(index, wind, load, seed=seed + 2)
    weather = generate_weather(index, seed=seed + 3)
    
    # Combine into feature DataFrame
    features = pd.DataFrame({
        "dam_eur_mwh": prices,
        "load_forecast_mw": load,
        "wind_total_mw": wind,
        **weather.to_dict(),
    })
    
    # Add calendar features
    features["hour"] = features.index.hour
    features["dow"] = features.index.dayofweek
    features["month"] = features.index.month
    features["is_weekend"] = (features["dow"] >= 5).astype(int)
    features["is_peak"] = features["hour"].between(7, 21).astype(int)
    
    # Add lags
    for lag in [1, 24, 48, 168]:
        features[f"price_lag_{lag}h"] = prices.shift(lag)
        features[f"load_lag_{lag}h"] = load.shift(lag)
    
    # Add target (next day price)
    features["target"] = prices.shift(-24)
    
    # Save if requested
    if save_to:
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        
        # Save prices as expected by the app
        prices_df = pd.DataFrame({"ts_utc": index, "dam_eur_mwh": prices.values})
        prices_df.to_parquet(save_to / "dam_prices.parquet", index=False)
        
        # Save full features
        features.to_parquet(save_to / "train.parquet")
        
        # Also save as CSV for inspection
        prices_df.head(100).to_csv(save_to / "dam_prices_sample.csv", index=False)
        
        print(f"✅ Saved synthetic data to {save_to}")
        print(f"   - {len(prices)} hours of data")
        print(f"   - Date range: {index.min()} to {index.max()}")
        print(f"   - Price range: €{prices.min():.2f} - €{prices.max():.2f}/MWh")
    
    return {
        "prices": prices,
        "load": load,
        "wind": wind,
        "weather": weather,
        "features": features,
    }


def generate_for_dashboard(days: int = 90, output_dir: str = "data") -> None:
    """
    Generate synthetic data in the format expected by the dashboard.
    
    This creates all the files your dashboard expects:
    - data/raw/dam_prices.parquet
    - data/processed/train.parquet
    """
    output_path = Path(output_dir)
    
    # Generate data
    data = generate_full_dataset(days=days, save_to=output_path / "processed")
    
    # Also save to raw directory for the DAM fetcher
    raw_dir = output_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    prices_df = pd.DataFrame({
        "ts_utc": data["prices"].index,
        "dam_eur_mwh": data["prices"].values
    })
    prices_df.to_parquet(raw_dir / "dam_prices.parquet", index=False)
    
    # Create a fake "Excel" file entry for the DAM cached reader
    prices_df.to_parquet(raw_dir / "dam_prices_cache.parquet", index=False)
    
    print("\n📊 Synthetic data ready for dashboard!")
    print(f"   Run: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--for-dashboard":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        generate_for_dashboard(days=days)
    else:
        # Demo generation
        print("Generating 30 days of synthetic Irish DAM data...")
        data = generate_full_dataset(days=30)
        
        print(f"\n📊 Generated data summary:")
        print(f"   Prices: min={data['prices'].min():.2f}, max={data['prices'].max():.2f}, mean={data['prices'].mean():.2f} €/MWh")
        print(f"   Load: min={data['load'].min():.0f}, max={data['load'].max():.0f}, mean={data['load'].mean():.0f} MW")
        print(f"   Wind: min={data['wind'].min():.0f}, max={data['wind'].max():.0f}, mean={data['wind'].mean():.0f} MW")
        
        print("\n💡 To generate data for your dashboard, run:")
        print("   python src/data/synthetic_generator.py --for-dashboard 90")
