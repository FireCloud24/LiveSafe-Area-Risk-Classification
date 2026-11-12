import numpy as np
import pandas as pd
import pytest

from forecast_evaluation import (
    evaluate_areas, forecast_last_value, forecast_linear_trend, forecast_seasonal_naive,
    rolling_origin_errors, summarise,
)


def test_last_value_repeats_the_final_month():
    assert list(forecast_last_value(np.array([1, 2, 7]), 3)) == [7, 7, 7]


def test_seasonal_naive_returns_the_same_months_last_year():
    history = np.arange(1, 15)  # months 1..14; month 15 one year earlier is month 3
    assert list(forecast_seasonal_naive(history, 3)) == [3, 4, 5]
    with pytest.raises(ValueError):
        forecast_seasonal_naive(np.arange(11), 1)


def test_linear_trend_extends_a_straight_line():
    history = 5 + 2 * np.arange(20)
    assert forecast_linear_trend(history, 2, window=10) == pytest.approx([45, 47])
    assert forecast_linear_trend(history, 2, window=None) == pytest.approx([45, 47])


def test_rolling_origin_never_uses_the_future():
    values = np.arange(30, dtype=float)
    seen = []

    def spy(history, horizon):
        seen.append(len(history))
        return np.zeros(horizon)

    errors = rolling_origin_errors(values, horizon=3, min_history=12, methods={"spy": spy})
    assert seen == list(range(12, 28))            # origins 12..27, each sees only the past
    assert len(errors) == 16 * 3
    first = errors.iloc[0]
    assert (first["origin"], first["step_ahead"], first["actual"]) == (12, 1, 12.0)


def test_seasonal_naive_is_perfect_on_a_purely_seasonal_series():
    pattern = np.array([5, 9, 4, 7, 8, 3, 6, 9, 2, 5, 7, 4], dtype=float)
    errors = rolling_origin_errors(np.tile(pattern, 3), horizon=3)
    mae = errors.groupby("method")["abs_error"].mean()
    assert mae["seasonal_naive"] == 0
    assert mae["linear_trend_10m"] > 0


def test_linear_trend_is_perfect_on_a_pure_trend():
    errors = rolling_origin_errors(100 + 3.0 * np.arange(30), horizon=3)
    mae = errors.groupby("method")["abs_error"].mean()
    assert mae["linear_trend_10m"] == pytest.approx(0, abs=1e-6)
    assert mae["seasonal_naive"] == pytest.approx(36)  # 12 months of a slope of 3


def test_gaps_are_rejected():
    with pytest.raises(ValueError):
        rolling_origin_errors([1.0, np.nan, 3.0] * 10)


def test_summarise_reports_one_row_per_area_with_a_best_method():
    months = pd.period_range("2022-01", periods=30, freq="M").astype(str)
    monthly = pd.concat([
        pd.DataFrame({"la_name": "Trend", "month": months, "total_crimes": 100 + 3.0 * np.arange(30)}),
        pd.DataFrame({"la_name": "Season", "month": months,
                      "total_crimes": np.tile([5, 9, 4, 7, 8, 3, 6, 9, 2, 5, 7, 4], 3)[:30]}),
    ])
    summary = summarise(evaluate_areas(monthly)).set_index("area")
    assert summary.loc["Trend", "best_method"] in {"linear_trend_10m", "linear_trend_all"}
    assert summary.loc["Season", "best_method"] == "seasonal_naive"
    assert summary.loc["Trend", "n_forecasts"] == 16 * 3


def test_backtest_joins_projections_to_recorded_months():
    from forecast_evaluation import backtest_projections, backtest_summary
    projections = pd.DataFrame({
        "area": ["A", "A", "B"], "months_ahead": [1, 2, 1],
        "month_label": ["2025-01", "2025-02", "2025-01"],
        "predicted_crimes": [110.0, 120.0, 50.0], "ci_lower": [90.0, 100.0, 40.0],
        "ci_upper": [130.0, 140.0, 60.0]})
    history = pd.DataFrame({"la_name": ["A", "A", "B"], "month": ["2025-01", "2024-12", "2025-01"],
                            "total_crimes": [100, 999, 75]})
    out = backtest_projections(projections, history)
    assert out["actual_crimes"].tolist()[0] == 100 and pd.isna(out["actual_crimes"].tolist()[1])
    assert out["abs_error"].tolist()[0] == 10 and out["abs_error"].tolist()[2] == 25
    assert out["inside_interval"].tolist()[0] is True or out["inside_interval"].tolist()[0] == True
    assert out["inside_interval"].tolist()[2] == False
    summary = backtest_summary(out)
    assert summary["n_projections"] == 3 and summary["n_scored"] == 2
    assert summary["n_inside_interval"] == 1 and summary["mae"] == 17.5
