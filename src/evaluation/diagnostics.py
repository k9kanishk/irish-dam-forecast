from __future__ import annotations

import pandas as pd


def analyze_errors_by_hour(results: pd.DataFrame) -> pd.DataFrame:
    """Analyze forecast errors by hour of day."""
    results = results.copy()
    results["hour"] = results.index.hour
    results["error"] = results["actual"] - results["predicted"]
    results["abs_error"] = results["error"].abs()

    return (
        results.groupby("hour")
        .agg({"error": ["mean", "std"], "abs_error": "mean", "actual": "mean"})
        .round(2)
    )


def analyze_errors_by_price_level(
    results: pd.DataFrame,
    bins: list[int] | None = None,
) -> pd.DataFrame:
    """Analyze errors by price level."""
    bins = bins or [0, 50, 100, 200, 500, 5000]
    results = results.copy()
    results["price_bin"] = pd.cut(results["actual"], bins=bins)
    results["error"] = results["actual"] - results["predicted"]
    results["abs_error"] = results["error"].abs()

    return (
        results.groupby("price_bin")
        .agg({"error": ["mean", "std"], "abs_error": "mean", "actual": "count"})
        .round(2)
    )
