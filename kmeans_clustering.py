import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path
from settings import PROCESSED_DIR

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
except ImportError:
    print("ERROR: Run  pip install scikit-learn")
    sys.exit(1)

INPUT_FILE   = PROCESSED_DIR / "area_features.parquet"
CLASS_FILE   = PROCESSED_DIR / "area_classified.parquet"
CLUSTER_FILE = PROCESSED_DIR / "area_clusters.parquet"
COMPARE_FILE = PROCESSED_DIR / "cluster_comparison.csv"

# Features used for clustering
CLUSTER_FEATURES = [
    "avg_monthly_crimes",
    "avg_violent_share",
    "avg_property_share",
    "crime_variability",
    "crime_trend_slope",
    "avg_monthly_prcp",
    "cold_month_count",
    "prcp_variability",
]

N_CLUSTERS = 4   # matches number of crime intensity levels


def find_optimal_k(X_scaled: np.ndarray, max_k: int = 8) -> None:
    """Print inertia and silhouette scores to help justify K choice."""
    print("\nElbow method — inertia and silhouette scores:")
    print(f"{'K':<5} {'Inertia':<12} {'Silhouette':<12}")
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil     = silhouette_score(X_scaled, labels)
        print(f"{k:<5} {inertia:<12.1f} {sil:<12.4f}")


def run_kmeans():
    print("Loading area features...")
    features   = pd.read_parquet(INPUT_FILE)
    classified = pd.read_parquet(CLASS_FILE)

    # Prepare feature matrix
    feat_cols = [c for c in CLUSTER_FEATURES if c in features.columns]
    X = features[feat_cols].copy()

    # Handle any remaining NaN
    X = X.fillna(X.median())

    # Standardise
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix: {X_scaled.shape[0]} areas × {X_scaled.shape[1]} features")
    print(f"Features used: {feat_cols}")

    # Find optimal K
    find_optimal_k(X_scaled)

    # Run K-Means with chosen K
    print(f"\nRunning K-Means with K={N_CLUSTERS}...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette score: {sil:.4f}  (range -1 to 1, higher = better separation)")

    # Assign meaningful cluster names based on centroid characteristics
    centroids_df = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=feat_cols
    )

    # Sort clusters by avg_monthly_crimes to assign interpretable names
    centroid_order = centroids_df["avg_monthly_crimes"].rank().astype(int) - 1
    cluster_name_map = {
        i: name for i, name in zip(
            centroids_df["avg_monthly_crimes"].argsort().values,
            ["Low Activity Cluster", "Moderate Activity Cluster",
             "Elevated Activity Cluster", "High Activity Cluster"]
        )
    }

    # Build results dataframe
    results = features[["la_code", "la_name"] + feat_cols].copy()
    results["cluster_id"]   = cluster_labels
    results["cluster_name"] = [cluster_name_map[c] for c in cluster_labels]

    # Merge with rule-based labels
    rule_cols = ["la_code", "overall_profile", "crime_intensity_label",
                 "crime_trend_label", "weather_exposure_label"]
    results = results.merge(
        classified[rule_cols], on="la_code", how="left"
    )

    results.to_parquet(CLUSTER_FILE, index=False)

    # Comparison table
    comparison = results[[
        "la_name",
        "cluster_name",
        "overall_profile",
        "crime_intensity_label",
        "avg_monthly_crimes",
        "avg_violent_share",
        "crime_trend_slope",
        "avg_monthly_prcp",
    ]].copy()

    comparison["avg_monthly_crimes"] = comparison["avg_monthly_crimes"].round(0).astype(int)
    comparison["avg_violent_share"]  = (comparison["avg_violent_share"] * 100).round(1)
    comparison["crime_trend_slope"]  = comparison["crime_trend_slope"].round(2)
    comparison["avg_monthly_prcp"]   = comparison["avg_monthly_prcp"].round(1)

    comparison = comparison.sort_values("cluster_name")
    comparison.to_csv(COMPARE_FILE, index=False)

    # Print comparison
    print("\n" + "="*80)
    print("CLUSTER vs RULE-BASED CLASSIFICATION COMPARISON")
    print("="*80)
    print(comparison.to_string(index=False))

    # Agreement analysis
    print("\n" + "="*80)
    print("CLUSTER COMPOSITION — which rule-based profiles land in each cluster?")
    print("="*80)
    for cname in sorted(results["cluster_name"].unique()):
        group = results[results["cluster_name"] == cname]
        print(f"\n{cname} ({len(group)} areas):")
        for _, row in group.iterrows():
            print(f"  {row['la_name']:<30} → {row['overall_profile']}")

    # Centroid summary
    print("\n" + "="*80)
    print("CLUSTER CENTROIDS (mean feature values per cluster)")
    print("="*80)
    centroid_summary = results.groupby("cluster_name")[feat_cols].mean().round(2)
    print(centroid_summary.to_string())

    # PCA for 2D visualisation data
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    results["pca_x"] = X_pca[:, 0].round(4)
    results["pca_y"] = X_pca[:, 1].round(4)
    results["pca_var_explained"] = round(sum(pca.explained_variance_ratio_) * 100, 1)

    results.to_parquet(CLUSTER_FILE, index=False)

    print(f"\nPCA variance explained by 2 components: {results['pca_var_explained'].iloc[0]}%")
    print(f"\nSaved:")
    print(f"  Clusters:    {CLUSTER_FILE}")
    print(f"  Comparison:  {COMPARE_FILE}")

    return results


if __name__ == "__main__":
    run_kmeans()
