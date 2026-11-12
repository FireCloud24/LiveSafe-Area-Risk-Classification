"""Rolling-origin evaluation of monthly crime forecasts against simple baselines.

A forecast is only worth reporting next to what a trivial rule would have scored.
Three trivial rules are used here:

- last value:       next month = this month
- seasonal naive:   next month = the same month one year earlier
- linear trend:     a straight line fitted to the most recent months (the project's model)

Rolling origin: stand at month t knowing only months before t, forecast the next few
months, compare with what happened, then move t forward one month and repeat. Every
forecast is scored on data the method had not seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEASON = 12


def forecast_last_value(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[-1], horizon).astype(float)


def forecast_seasonal_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    if len(history) < SEASON:
        raise ValueError("seasonal naive needs at least 12 months of history")
    return np.array([history[len(history) + h - SEASON] for h in range(horizon)], dtype=float)


def forecast_linear_trend(history: np.ndarray, horizon: int, window: int | None = 10) -> np.ndarray:
    """Straight line through the last `window` months (all months if window is None)."""
    recent = history if window is None else history[-window:]
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent, 1)
    future_x = np.arange(len(recent), len(recent) + horizon)
    return slope * future_x + intercept


METHODS = {
    "linear_trend_10m": lambda history, horizon: forecast_linear_trend(history, horizon, 10),
    "linear_trend_all": lambda history, horizon: forecast_linear_trend(history, horizon, None),
    "last_value": forecast_last_value,
    "seasonal_naive": forecast_seasonal_naive,
}


def rolling_origin_errors(series, horizon: int = 3, min_history: int = SEASON,
                          methods: dict | None = None) -> pd.DataFrame:
    """One row per (origin, step ahead, method) with the absolute error.

    `series` is a 1-D sequence of monthly values in time order with no gaps.
    The first origin is `min_history`, so every method has at least that many months.
    Only origins with a full `horizon` of actual values ahead are used, so every method
    and every step ahead is scored on the same months.
    """
    values = np.asarray(series, dtype=float)
    if np.isnan(values).any():
        raise ValueError("series has gaps; fill or trim it before evaluating")
    methods = methods or METHODS
    rows = []
    for origin in range(min_history, len(values) - horizon + 1):
        history, actual = values[:origin], values[origin:origin + horizon]
        for name, method in methods.items():
            predicted = method(history, horizon)
            for step in range(horizon):
                rows.append({"origin": origin, "step_ahead": step + 1, "method": name,
                             "actual": actual[step], "predicted": float(predicted[step]),
                             "abs_error": abs(actual[step] - float(predicted[step]))})
    return pd.DataFrame(rows)


def evaluate_areas(monthly: pd.DataFrame, value: str = "total_crimes", horizon: int = 3,
                   min_history: int = SEASON) -> pd.DataFrame:
    """Rolling-origin errors for every area in a long table (la_name, month, value)."""
    frames = []
    for la_name, group in monthly.sort_values("month").groupby("la_name"):
        errors = rolling_origin_errors(group[value].to_numpy(), horizon, min_history)
        errors.insert(0, "area", la_name)
        errors["mean_monthly"] = group[value].mean()
        frames.append(errors)
    return pd.concat(frames, ignore_index=True)


def summarise(errors: pd.DataFrame) -> pd.DataFrame:
    """MAE per area and method (wide), plus MAE as a percentage of the area's mean month."""
    mae = errors.pivot_table(index="area", columns="method", values="abs_error", aggfunc="mean")
    mean_monthly = errors.groupby("area")["mean_monthly"].first()
    pct = mae.div(mean_monthly, axis=0) * 100
    out = mae.round(1).add_prefix("mae_")
    out = out.join(pct.round(2).add_prefix("mae_pct_"))
    out["n_forecasts"] = errors.groupby("area").apply(
        lambda g: len(g) // g["method"].nunique(), include_groups=False)
    out["best_method"] = mae.idxmin(axis=1)
    return out.reset_index()
