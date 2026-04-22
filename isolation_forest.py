import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Paths (matching your settings.py conventions)
INPUT_PATH = Path("data/processed/neighbourhood_features.parquet")
OUTPUT_PATH = Path("data/processed/neighbourhood_anomalies.parquet")
SUMMARY_PATH = Path("data/processed/anomaly_summary.csv")

# Configuration
CONTAMINATION = 0.08   # Expect ~8% of neighbourhoods to be anomalous
RANDOM_STATE = 42
N_ESTIMATORS = 200     # Number of isolation trees

# Features to use for anomaly detection.
FEATURE_COLS_CANDIDATES = [
    ["avg_monthly_crimes", "avg_monthly_crimes"],
    ["violent_share",      "avg_violent_share"],
    ["property_share",     "avg_property_share"],
    ["trend_slope",        "crime_trend_slope"],
    ["crime_variability",  "crime_variability"],
]

def _resolve_feature_cols(df):
    selected = []
    for preferred, fallback in FEATURE_COLS_CANDIDATES:
        if preferred in df.columns:
            selected.append(preferred)
        elif fallback in df.columns:
            selected.append(fallback)
    return selected


def load_data():
    """Load neighbourhood features."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run build_neighbourhood_features.py first."
        )
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} neighbourhoods from {INPUT_PATH}")
    return df


def detect_anomalies_global(df):
    # Resolve feature columns against actual dataframe columns
    available_features = _resolve_feature_cols(df)
    if len(available_features) < 2:
        raise ValueError(
            f"Need at least 2 feature columns. Found: {available_features}\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"Using features: {available_features}")

    X = df[available_features].copy()

    # Handle any missing values
    X = X.fillna(X.median())

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)

    predictions = iso_forest.predict(X_scaled)

    raw_scores = iso_forest.decision_function(X_scaled)
    anomaly_scores = -raw_scores

    score_min = anomaly_scores.min()
    score_max = anomaly_scores.max()
    if score_max > score_min:
        anomaly_scores_norm = (anomaly_scores - score_min) / (score_max - score_min)
    else:
        anomaly_scores_norm = np.zeros_like(anomaly_scores)

    df = df.copy()
    df["anomaly_score"] = anomaly_scores_norm.round(4)
    df["is_anomaly"] = (predictions == -1).astype(int)

    return df


def detect_anomalies_within_la(df):
    """
    Also run Isolation Forest WITHIN each LA, to find neighbourhoods
    that are anomalous relative to their own local authority.
    This is useful because a neighbourhood might be normal nationally
    but unusual for its specific LA.
    """
    available_features = _resolve_feature_cols(df)
    df = df.copy()
    df["is_anomaly_within_la"] = 0
    df["anomaly_score_within_la"] = 0.0

    la_col = None
    for candidate in ["la_name", "la_code", "lad24nm", "lad24cd", "force"]:
        if candidate in df.columns:
            la_col = candidate
            break

    if la_col is None:
        print("Warning: No LA identifier column found. Skipping within-LA analysis.")
        return df

    print(f"\nWithin-LA anomaly detection (grouping by '{la_col}'):")

    for la_name, group in df.groupby(la_col):
        if len(group) < 10:
            # Too few neighbourhoods for meaningful anomaly detection
            print(f"  {la_name}: {len(group)} neighbourhoods (skipped — too few)")
            continue

        X = group[available_features].fillna(group[available_features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        local_contamination = min(CONTAMINATION, 2 / len(group))

        iso = IsolationForest(
            n_estimators=100,
            contamination=max(local_contamination, 0.01),
            random_state=RANDOM_STATE,
        )
        iso.fit(X_scaled)

        preds = iso.predict(X_scaled)
        scores = -iso.decision_function(X_scaled)

        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores_norm = (scores - s_min) / (s_max - s_min)
        else:
            scores_norm = np.zeros_like(scores)

        df.loc[group.index, "is_anomaly_within_la"] = (preds == -1).astype(int)
        df.loc[group.index, "anomaly_score_within_la"] = scores_norm.round(4)

        n_anomalies = (preds == -1).sum()
        print(f"  {la_name}: {len(group)} neighbourhoods, {n_anomalies} anomalies")

    return df


def create_summary(df):
    """Create a human-readable summary of anomalies."""
    la_col = None
    for candidate in ["la_name", "la_code", "lad24nm", "lad24cd", "force"]:
        if candidate in df.columns:
            la_col = candidate
            break

    # Identify name column for neighbourhood
    name_col = None
    for candidate in ["neighbourhood_name", "name", "neighbourhood", "nhood_name", "id"]:
        if candidate in df.columns:
            name_col = candidate
            break

    # Global anomalies summary
    anomalies = df[df["is_anomaly"] == 1].copy()
    print(f"\n{'='*60}")
    print(f"ANOMALY DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Total neighbourhoods analysed: {len(df)}")
    print(f"Global anomalies detected: {len(anomalies)} ({100*len(anomalies)/len(df):.1f}%)")

    if la_col:
        print(f"\nAnomalies by local authority:")
        summary = (
            df.groupby(la_col)
            .agg(
                total_neighbourhoods=("is_anomaly", "count"),
                global_anomalies=("is_anomaly", "sum"),
                within_la_anomalies=("is_anomaly_within_la", "sum"),
                mean_anomaly_score=("anomaly_score", "mean"),
            )
            .reset_index()
        )
        summary["global_anomaly_pct"] = (
            100 * summary["global_anomalies"] / summary["total_neighbourhoods"]
        ).round(1)
        summary = summary.sort_values("global_anomalies", ascending=False)
        print(summary.to_string(index=False))
        return summary

    return None


def main():
    print("=" * 60)
    print("ISOLATION FOREST — Neighbourhood Anomaly Detection")
    print("=" * 60)

    # Load data
    df = load_data()
    print(f"Columns available: {list(df.columns)}")

    # Global anomaly detection
    print("\n--- Global Anomaly Detection ---")
    df = detect_anomalies_global(df)

    # Within-LA anomaly detection
    print("\n--- Within-LA Anomaly Detection ---")
    df = detect_anomalies_within_la(df)

    # Create summary
    summary = create_summary(df)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved neighbourhood anomalies to {OUTPUT_PATH}")

    if summary is not None:
        summary.to_csv(SUMMARY_PATH, index=False)
        print(f"Saved anomaly summary to {SUMMARY_PATH}")

    # Print top 10 most anomalous neighbourhoods
    name_col = None
    for candidate in ["neighbourhood_name", "name", "neighbourhood", "nhood_name", "id"]:
        if candidate in df.columns:
            name_col = candidate
            break

    print(f"\nTop 10 most anomalous neighbourhoods (globally):")
    top = df.nlargest(10, "anomaly_score")
    display_cols = [c for c in [name_col, "anomaly_score", "is_anomaly", "avg_monthly_crimes"] if c and c in top.columns]
    if display_cols:
        print(top[display_cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
