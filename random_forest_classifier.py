import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
FEATURES_PATH = Path("data/processed/neighbourhood_features.parquet")
CLASSIFIED_PATH = Path("data/processed/neighbourhood_classified.parquet")
OUTPUT_DIR = Path("data/processed")

# Configuration
RANDOM_STATE = 42
N_ESTIMATORS = 300
CV_FOLDS = 5

# Features to use for classification.
_FEATURE_CANDIDATES = [
    ["avg_monthly_crimes", "avg_monthly_crimes"],
    ["violent_share",      "avg_violent_share"],
    ["property_share",     "avg_property_share"],
    ["trend_slope",        "crime_trend_slope"],
    ["crime_variability",  "crime_variability"],
]

def _resolve_features(df):
    selected = []
    for preferred, fallback in _FEATURE_CANDIDATES:
        if preferred in df.columns:
            selected.append(preferred)
        elif fallback in df.columns:
            selected.append(fallback)
    return selected

TARGET_CANDIDATES = [
    "neigh_intensity",
    "crime_intensity",
    "crime_level",
    "neigh_class",
    "classification",
    "crime_category",
    "crime_class",
    "intensity",
]


def load_data():
    """Load neighbourhood data with classification labels."""
    for path in [CLASSIFIED_PATH, FEATURES_PATH]:
        if path.exists():
            df = pd.read_parquet(path)
            print(f"Loaded {len(df)} rows from {path}")
            print(f"Columns: {list(df.columns)}")
            return df

    raise FileNotFoundError(
        "Neither neighbourhood_classified.parquet nor "
        "neighbourhood_features.parquet found in data/processed/"
    )


def find_target_column(df):
    """Find the classification label column."""
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            print(f"Using target column: '{col}'")
            return col

    for col in df.columns:
        if df[col].dtype == "object":
            unique_vals = df[col].nunique()
            if 2 <= unique_vals <= 5:
                vals = df[col].unique()
                print(f"Candidate target column: '{col}' with values {vals}")
                return col

    raise ValueError(
        f"Could not find target column. Tried: {TARGET_CANDIDATES}\n"
        f"Available columns: {list(df.columns)}\n"
        f"Hint: Add the column name to TARGET_CANDIDATES in this script."
    )


def prepare_data(df, target_col):
    """Prepare features and target for classification."""
    available_features = _resolve_features(df)

    if len(available_features) < 2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {
            "rank", "crime_rank", "la_rank", "index",
            "latitude", "longitude", "lat", "lng",
            "centroid_lat", "centroid_lon",
            "total_crimes", "violent_count", "property_count",
            "months_present",
        }
        available_features = [c for c in numeric_cols if c.lower() not in exclude]
        print(f"Auto-detected numeric features: {available_features}")

    print(f"Features used ({len(available_features)}): {available_features}")

    X = df[available_features].copy()
    y = df[target_col].copy()

    # Drop rows with missing target
    valid_mask = y.notna() & ~y.isin(["", "Unknown", "N/A"])
    X = X[valid_mask]
    y = y[valid_mask]

    # Fill missing feature values with median
    X = X.fillna(X.median())

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    print(f"\nTarget classes: {list(class_names)}")
    print(f"Class distribution:")
    for cls in class_names:
        count = (y == cls).sum()
        print(f"  {cls}: {count} ({100*count/len(y):.1f}%)")

    return X, y_encoded, class_names, available_features, le


def train_and_evaluate(X, y, class_names, feature_names):
    """
    Train Random Forest with stratified k-fold cross-validation.
    Returns the trained model and evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"RANDOM FOREST CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}, Features: {len(feature_names)}, Folds: {CV_FOLDS}")

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create model
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Cross-validated predictions
    y_pred = cross_val_predict(rf, X_scaled, y, cv=cv)

    # Cross-validated accuracy scores per fold
    fold_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")

    print(f"\nCross-Validation Results ({CV_FOLDS}-fold):")
    print(f"  Accuracy per fold: {[f'{s:.3f}' for s in fold_scores]}")
    print(f"  Mean accuracy:     {fold_scores.mean():.3f} ± {fold_scores.std():.3f}")

    # Full classification report
    report_str = classification_report(y, y_pred, target_names=class_names)
    print(f"\nClassification Report:\n{report_str}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    # Train final model on all data for feature importance
    rf.fit(X_scaled, y)

    # Feature importance
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "importance_pct": (importances * 100).round(2),
    }).sort_values("importance", ascending=False)

    print(f"\nFeature Importance (Mean Decrease in Impurity):")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance_pct"])
        print(f"  {row['feature']:25s} {row['importance_pct']:5.1f}%  {bar}")

    # Parse classification report into DataFrame
    report_dict = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).T

    return rf, y_pred, fold_scores, report_df, cm_df, importance_df


def save_results(df_original, y_pred, le, fold_scores, report_df, cm_df, importance_df):
    """Save all results to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Classification report
    report_path = OUTPUT_DIR / "rf_classification_report.csv"
    report_df.to_csv(report_path)
    print(f"\nSaved classification report to {report_path}")

    # Feature importance
    importance_path = OUTPUT_DIR / "rf_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to {importance_path}")

    # Confusion matrix
    cm_path = OUTPUT_DIR / "rf_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Predictions with probabilities
    result_df = df_original.copy()

    pred_labels = le.inverse_transform(y_pred)
    
    pred_df = pd.DataFrame({
        "predicted_label": pred_labels,
    })
    pred_path = OUTPUT_DIR / "rf_predictions.parquet"

    # Save fold scores
    scores_df = pd.DataFrame({
        "fold": range(1, len(fold_scores) + 1),
        "accuracy": fold_scores,
    })
    scores_path = OUTPUT_DIR / "rf_fold_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved fold scores to {scores_path}")

    # Save a summary text file for easy reference
    summary_path = OUTPUT_DIR / "rf_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Random Forest Classification Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(y_pred)}\n")
        f.write(f"CV Folds: {len(fold_scores)}\n")
        f.write(f"Mean accuracy: {fold_scores.mean():.4f}\n")
        f.write(f"Std accuracy:  {fold_scores.std():.4f}\n\n")
        f.write("Feature Importance:\n")
        for _, row in importance_df.iterrows():
            f.write(f"  {row['feature']}: {row['importance_pct']:.1f}%\n")
    print(f"Saved summary to {summary_path}")


def main():
    print("=" * 60)
    print("RANDOM FOREST — Supervised Crime Category Classification")
    print("=" * 60)

    # Load data
    df = load_data()

    # Find target column
    target_col = find_target_column(df)

    # Prepare data
    X, y, class_names, feature_names, le = prepare_data(df, target_col)

    if len(X) < 30:
        print(f"\nWarning: Only {len(X)} samples. Results may not be reliable.")
        print("This technique works best with the neighbourhood dataset (1500+ rows).")

    # Train and evaluate
    rf, y_pred, fold_scores, report_df, cm_df, importance_df = train_and_evaluate(
        X, y, class_names, feature_names
    )

    # Save results
    save_results(df, y_pred, le, fold_scores, report_df, cm_df, importance_df)

    print(f"\n{'='*60}")
    print("KEY TAKEAWAYS FOR YOUR REPORT:")
    print(f"{'='*60}")
    print(f"1. Mean cross-validated accuracy: {fold_scores.mean():.1%}")
    print(f"2. Top feature: {importance_df.iloc[0]['feature']} "
          f"({importance_df.iloc[0]['importance_pct']:.1f}%)")
    print(f"3. If accuracy is high (>80%), your rule-based labels capture")
    print(f"   genuine patterns in the data that ML can also learn.")
    print(f"4. Feature importance shows which variables drive classification")
    print(f"   — compare this to your rule-based threshold choices.")
    print("\nDone.")


if __name__ == "__main__":
    main()
