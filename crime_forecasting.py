import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
INPUT_PATH = Path("data/processed/crime_monthly.parquet")
OUTPUT_DIR = Path("data/processed")

# Configuration
TRAIN_MONTHS = 10
TEST_MONTHS = 2
FORECAST_AHEAD = 3
CONFIDENCE_LEVEL = 0.95

# Column names
LA_CODE_COL = "la_code"
LA_NAME_COL = "la_name"
MONTH_COL = "month"
CRIME_COL = "total_crimes"


def load_data():
    """Load monthly crime data."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run build_features.py first."
        )
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    return df


def detect_columns(df):
    """Auto-detect column names."""
    # LA identifier
    la_col = None
    for candidate in [LA_CODE_COL, LA_NAME_COL, "lad24cd", "lad24nm", "la_code", "la_name", "area"]:
        if candidate in df.columns:
            la_col = candidate
            break
    if la_col is None:
        raise ValueError(f"No LA identifier column found. Columns: {list(df.columns)}")

    # Month column
    month_col = None
    for candidate in [MONTH_COL, "month", "month_num", "month_index"]:
        if candidate in df.columns:
            month_col = candidate
            break

    # If no month column, try to extract from date
    if month_col is None:
        for candidate in ["date", "year_month", "period"]:
            if candidate in df.columns:
                try:
                    df["month"] = pd.to_datetime(df[candidate]).dt.month
                    month_col = "month"
                    break
                except Exception:
                    continue

    if month_col is None:
        raise ValueError(f"No month column found. Columns: {list(df.columns)}")

    # Crime count column
    crime_col = None
    for candidate in [CRIME_COL, "total_crimes", "crime_count", "count", "crimes"]:
        if candidate in df.columns:
            crime_col = candidate
            break
    if crime_col is None:
        raise ValueError(f"No crime count column found. Columns: {list(df.columns)}")

    # Get LA name column for display
    name_col = la_col
    for candidate in [LA_NAME_COL, "la_name", "lad24nm", "name"]:
        if candidate in df.columns:
            name_col = candidate
            break

    print(f"Using columns: LA={la_col}, month={month_col}, crimes={crime_col}, name={name_col}")
    return la_col, month_col, crime_col, name_col


def forecast_area(area_data, month_col, crime_col, area_name):

    area_data = area_data.copy()

    try:
        area_data["_sort_key"] = pd.to_datetime(area_data[month_col], errors="coerce")
        if area_data["_sort_key"].isna().all():
            area_data["_sort_key"] = pd.to_numeric(area_data[month_col], errors="coerce")
    except Exception:
        area_data["_sort_key"] = area_data[month_col]

    area_data = area_data.dropna(subset=["_sort_key"])
    area_data = area_data.sort_values("_sort_key").reset_index(drop=True)
    area_data = area_data.drop(columns=["_sort_key"])

    if area_data.empty:
        return None, None, None

    area_data["month_index"] = np.arange(len(area_data))

    train = area_data[area_data["month_index"] < TRAIN_MONTHS]
    test  = area_data[area_data["month_index"] >= TRAIN_MONTHS]

    if len(train) < 3:
        return None, None, None

    X_train = train[["month_index"]].values
    y_train = train[crime_col].values
    X_test = test[["month_index"]].values if len(test) > 0 else np.array([]).reshape(-1, 1)
    y_test = test[crime_col].values if len(test) > 0 else np.array([])

    # Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Training predictions
    y_train_pred = model.predict(X_train)
    train_residuals = y_train - y_train_pred
    residual_std = np.std(train_residuals, ddof=1) if len(train_residuals) > 1 else 0

    # Test predictions
    metrics = {}
    if len(X_test) > 0 and len(y_test) > 0:
        y_test_pred = model.predict(X_test)
        metrics["mae"] = mean_absolute_error(y_test, y_test_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, y_test_pred))
        metrics["r2_train"] = r2_score(y_train, y_train_pred)
        # MAPE — handle division by zero
        nonzero_mask = y_test != 0
        if nonzero_mask.any():
            metrics["mape"] = np.mean(
                np.abs((y_test[nonzero_mask] - y_test_pred[nonzero_mask]) / y_test[nonzero_mask])
            ) * 100
        else:
            metrics["mape"] = np.nan
    else:
        metrics["mae"] = np.nan
        metrics["rmse"] = np.nan
        metrics["r2_train"] = r2_score(y_train, y_train_pred) if len(y_train) > 1 else np.nan
        metrics["mape"] = np.nan

    metrics["slope"] = model.coef_[0]
    metrics["intercept"] = model.intercept_
    metrics["residual_std"] = residual_std
    metrics["area"] = area_name

    # Trend interpretation
    if model.coef_[0] > 0.5:
        metrics["trend"] = "Increasing"
    elif model.coef_[0] < -0.5:
        metrics["trend"] = "Decreasing"
    else:
        metrics["trend"] = "Stable"

    # Forward projections
    max_index = area_data["month_index"].max()
    future_indices = np.array([[max_index + i] for i in range(1, FORECAST_AHEAD + 1)])
    future_pred = model.predict(future_indices)

    # Confidence interval using t-distribution
    from scipy import stats
    t_val = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, df=max(len(y_train) - 2, 1))

    projections = []
    for i, (idx, pred) in enumerate(zip(future_indices.flatten(), future_pred)):
        ci = t_val * residual_std * np.sqrt(1 + 1/len(X_train) +
              (idx - X_train.mean())**2 / np.sum((X_train - X_train.mean())**2))
        projections.append({
            "area": area_name,
            "months_ahead": i + 1,
            "month_index": int(idx),
            "predicted_crimes": max(0, round(pred, 1)),
            "ci_lower": max(0, round(pred - ci, 1)),
            "ci_upper": round(pred + ci, 1),
        })

    # Build full prediction series for this area
    all_indices = np.concatenate([X_train.flatten(), X_test.flatten() if len(X_test) > 0 else []])
    all_actuals = np.concatenate([y_train, y_test])
    all_preds = model.predict(all_indices.reshape(-1, 1))
    all_splits = (["train"] * len(X_train)) + (["test"] * len(X_test))

    pred_df = pd.DataFrame({
        "area": area_name,
        "month_index": all_indices.astype(int),
        "actual": all_actuals,
        "predicted": all_preds.round(1),
        "split": all_splits,
    })

    return metrics, projections, pred_df


def main():
    print("=" * 60)
    print("CRIME TREND FORECASTING — Linear Regression")
    print("=" * 60)

    # Load data
    df = load_data()

    # Detect columns
    la_col, month_col, crime_col, name_col = detect_columns(df)

    # Process each area
    all_metrics = []
    all_projections = []
    all_predictions = []

    areas = df[la_col].unique()
    print(f"\nProcessing {len(areas)} areas...")

    for area_id in sorted(areas):
        area_data = df[df[la_col] == area_id].copy()
        area_name = area_data[name_col].iloc[0] if name_col != la_col else area_id

        if len(area_data) < 6:
            print(f"  {area_name}: skipped (only {len(area_data)} months)")
            continue

        metrics, projections, pred_df = forecast_area(
            area_data, month_col, crime_col, area_name
        )

        if metrics is None:
            print(f"  {area_name}: skipped (insufficient training data)")
            continue

        all_metrics.append(metrics)
        all_projections.extend(projections)
        all_predictions.append(pred_df)

        mae_str = f"MAE={metrics['mae']:.1f}" if not np.isnan(metrics['mae']) else "MAE=N/A"
        print(f"  {area_name:25s} slope={metrics['slope']:+.2f}  {mae_str}  → {metrics['trend']}")

    # Combine results
    metrics_df = pd.DataFrame(all_metrics)
    projections_df = pd.DataFrame(all_projections)
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    if not metrics_df.empty and "area" in metrics_df.columns:
        classified_path = Path("data/processed/area_classified.parquet")
        if classified_path.exists():
            try:
                cdf = pd.read_parquet(classified_path)
                if "la_code" in cdf.columns and "la_name" in cdf.columns:
                    code_to_name = dict(zip(cdf["la_code"], cdf["la_name"]))
                    sample = str(metrics_df["area"].iloc[0])
                    if sample.startswith(("E", "W")) and len(sample) == 9:
                        metrics_df["area"]    = metrics_df["area"].map(code_to_name).fillna(metrics_df["area"])
                        projections_df["area"] = projections_df["area"].map(code_to_name).fillna(projections_df["area"])
                        predictions_df["area"] = predictions_df["area"].map(code_to_name).fillna(predictions_df["area"]) if not predictions_df.empty else predictions_df
                        print("Mapped LA codes to readable names.")
            except Exception as e:
                print(f"Note: Could not map LA codes to names: {e}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("FORECASTING SUMMARY")
    print(f"{'='*60}")
    if not metrics_df.empty:
        valid_mae = metrics_df["mae"].dropna()
        valid_rmse = metrics_df["rmse"].dropna()
        if len(valid_mae) > 0:
            print(f"Mean MAE across areas:  {valid_mae.mean():.1f}")
            print(f"Mean RMSE across areas: {valid_rmse.mean():.1f}")
        print(f"Mean slope:             {metrics_df['slope'].mean():+.2f}")
        print(f"\nTrend distribution:")
        print(metrics_df["trend"].value_counts().to_string())

        print(f"\n3-Month Projections:")
        for area in metrics_df["area"]:
            area_proj = projections_df[projections_df["area"] == area]
            if not area_proj.empty:
                last = area_proj.iloc[-1]
                print(
                    f"  {area:25s} → {last['predicted_crimes']:.0f} crimes/month "
                    f"(95% CI: {last['ci_lower']:.0f}–{last['ci_upper']:.0f})"
                )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not metrics_df.empty:
        metrics_path = OUTPUT_DIR / "forecast_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nSaved metrics to {metrics_path}")

    if not projections_df.empty:
        proj_path = OUTPUT_DIR / "forecast_projections.csv"
        projections_df.to_csv(proj_path, index=False)
        print(f"Saved projections to {proj_path}")

    if not predictions_df.empty:
        pred_path = OUTPUT_DIR / "crime_forecasts.parquet"
        predictions_df.to_parquet(pred_path, index=False)
        print(f"Saved full predictions to {pred_path}")

    print(f"\n{'='*60}")
    print("KEY TAKEAWAYS FOR YOUR REPORT:")
    print(f"{'='*60}")
    print("1. Linear regression provides a proper predictive model with")
    print("   measurable accuracy (MAE, RMSE), not just a descriptive slope.")
    print("2. Train on Jan-Oct, test on Nov-Dec = genuine holdout evaluation.")
    print("3. Forward projections with confidence intervals show uncertainty.")
    print("4. Compare MAE to mean crime count — if MAE is <20% of mean,")
    print("   the model has reasonable predictive power.")
    print("5. Wide confidence intervals honestly reflect the limitation of")
    print("   extrapolating from 12 months of data.")
    print("\nDone.")


if __name__ == "__main__":
    main()
