import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error (with floor to avoid tiny denominators)."""
    return float(100 * np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(100 * np.mean(numerator / np.maximum(denominator, 1e-6)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball loss for quantile forecasts."""
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction interval coverage."""
    within = (y_true >= lower) & (y_true <= upper)
    return float(within.mean())


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average prediction interval width."""
    return float(np.mean(upper - lower))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    """Compute all relevant metrics."""
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }

    if lower is not None and upper is not None:
        metrics["coverage_90"] = coverage(y_true, lower, upper)
        metrics["interval_width"] = interval_width(lower, upper)

    return metrics
