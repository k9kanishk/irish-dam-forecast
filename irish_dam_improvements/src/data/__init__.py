# src/data/__init__.py
"""
Data fetching and generation modules for Irish DAM Forecast.
"""
from .synthetic_generator import generate_full_dataset, generate_for_dashboard

__all__ = [
    "generate_full_dataset",
    "generate_for_dashboard",
]
