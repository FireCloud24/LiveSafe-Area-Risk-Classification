import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
CLASSIFIED_PATH = Path("data/processed/area_classified.parquet")
IMD_PATH = Path("data/external/imd_scores.csv")
OUTPUT_DIR = Path("data/processed")

FALLBACK_AVAILABLE = True

IMD_FALLBACK = {
    # LA Code: (LA Name, approx IMD crime domain avg score, source)
    "E08000019": ("Sheffield", 0.42, "IoD2019"),
    "E08000035": ("Leeds", 0.39, "IoD2019"),
    "E06000023": ("Bristol, City of", 0.35, "IoD2019"),
    "E08000026": ("Coventry", 0.30, "IoD2019"),
    "E08000021": ("Newcastle upon Tyne", 0.40, "IoD2019"),
    "E06000016": ("Leicester", 0.45, "IoD2019"),
    "E07000148": ("Norwich", 0.20, "IoD2019"),
    "E07000041": ("Exeter", 0.15, "IoD2019"),
    "E06000047": ("County Durham", 0.22, "IoD2019"),
    "E06000054": ("Wiltshire", 0.05, "IoD2019"),
    "E06000019": ("Herefordshire, County of", 0.04, "IoD2019"),
    "E06000065": ("North Yorkshire", 0.06, "IoD2019"),
    "E06000052": ("Cornwall", 0.08, "IoD2019"),
    # Welsh areas — WIMD uses different scale, approximate equivalents
    "W06000015": ("Cardiff", 0.33, "WIMD2019"),
    "W06000011": ("Swansea", 0.25, "WIMD2019"),
}

# Crime intensity label ordering (for Spearman correlation)
CRIME_INTENSITY_ORDER = {
    "Very Low Recorded Crime": 1,
    "Moderate Crime Level": 2,
    "Elevated Crime Level": 3,
    "Severe Crime Concentration": 4,
}


def load_classified_areas():
    """Load the classified areas from your pipeline."""
    if not CLASSIFIED_PATH.exists():
        raise FileNotFoundError(
            f"{CLASSIFIED_PATH} not found. Run classify_areas.py first."
        )
    df = pd.read_parquet(CLASSIFIED_PATH)
    print(f"Loaded {len(df)} classified areas")
    print(f"Columns: {list(df.columns)}")
    return df


def find_columns(df):
    """Auto-detect relevant columns."""
    la_col = None
    for c in ["la_code", "lad24cd", "code"]:
        if c in df.columns:
            la_col = c
            break

    name_col = None
    for c in ["la_name", "lad24nm", "name"]:
        if c in df.columns:
            name_col = c
            break

    intensity_col = None
    for c in ["crime_intensity", "crime_level", "intensity"]:
        if c in df.columns:
            intensity_col = c
            break

    crimes_col = None
    for c in ["avg_monthly_crimes", "total_crimes_12m", "total_crimes"]:
        if c in df.columns:
            crimes_col = c
            break

    return la_col, name_col, intensity_col, crimes_col


def load_imd_data(la_codes):
    """
    Load IMD data — try full file first, then fallback.
    Returns DataFrame with la_code and imd_crime_score columns.
    """
    if IMD_PATH.exists():
        print(f"\nLoading IMD data from {IMD_PATH}...")
        imd = pd.read_csv(IMD_PATH)
        print(f"IMD columns: {list(imd.columns)[:10]}...")  # First 10

        # Try to find LA code column
        la_code_col = None
        for c in imd.columns:
            if "local authority" in c.lower() and "code" in c.lower():
                la_code_col = c
                break
            if "lad" in c.lower() and "cd" in c.lower():
                la_code_col = c
                break

        # Try to find crime domain score
        crime_score_col = None
        for c in imd.columns:
            if "crime" in c.lower() and ("score" in c.lower() or "rate" in c.lower()):
                crime_score_col = c
                break

        if la_code_col and crime_score_col:
            print(f"Using columns: LA={la_code_col}, Crime Score={crime_score_col}")
            imd_la = (
                imd.groupby(la_code_col)[crime_score_col]
                .mean()
                .reset_index()
                .rename(columns={la_code_col: "la_code", crime_score_col: "imd_crime_score"})
            )
            imd_la = imd_la[imd_la["la_code"].isin(la_codes)]
            if len(imd_la) > 0:
                print(f"Found IMD scores for {len(imd_la)} of {len(la_codes)} areas")
                return imd_la
            else:
                print("No matching LA codes found in IMD file. Using fallback.")

    # Fallback to built-in data
    if FALLBACK_AVAILABLE:
        print("\nUsing fallback IMD data (built-in approximate values)")
        print("NOTE: Replace these with actual IMD values for your report!")
        rows = []
        for code, (name, score, source) in IMD_FALLBACK.items():
            if code in la_codes:
                rows.append({"la_code": code, "la_name": name,
                            "imd_crime_score": score, "imd_source": source})
        return pd.DataFrame(rows)
    else:
        raise FileNotFoundError(
            f"No IMD data available. Either:\n"
            f"1. Download IoD2019 CSV from gov.uk and place at {IMD_PATH}\n"
            f"2. Set FALLBACK_AVAILABLE = True and verify the fallback values"
        )


def compute_correlations(comparison_df):
    results = {}

    # Spearman rank correlation between crime intensity rank and IMD rank
    if "intensity_rank" in comparison_df.columns and "imd_rank" in comparison_df.columns:
        spearman_r, spearman_p = stats.spearmanr(
            comparison_df["intensity_rank"],
            comparison_df["imd_rank"]
        )
        results["spearman_r"] = spearman_r
        results["spearman_p"] = spearman_p

        kendall_tau, kendall_p = stats.kendalltau(
            comparison_df["intensity_rank"],
            comparison_df["imd_rank"]
        )
        results["kendall_tau"] = kendall_tau
        results["kendall_p"] = kendall_p

    if "avg_monthly_crimes" in comparison_df.columns:
        r_crimes, p_crimes = stats.spearmanr(
            comparison_df["avg_monthly_crimes"],
            comparison_df["imd_crime_score"]
        )
        results["spearman_crimes_vs_imd"] = r_crimes
        results["spearman_crimes_vs_imd_p"] = p_crimes

    return results


def main():
    print("=" * 60)
    print("IMD VALIDATION — External Ground-Truth Comparison")
    print("=" * 60)

    # Load classified areas
    df = load_classified_areas()
    la_col, name_col, intensity_col, crimes_col = find_columns(df)

    if la_col is None:
        raise ValueError("Cannot find LA code column in classified data.")

    la_codes = set(df[la_col].unique())
    print(f"LAs in classified data: {len(la_codes)}")

    # Load IMD data
    imd_df = load_imd_data(la_codes)

    # Merge
    comparison = df.merge(imd_df, left_on=la_col, right_on="la_code", how="inner")
    print(f"\nMatched {len(comparison)} areas for comparison")

    if len(comparison) < 5:
        print("Warning: Very few matched areas. Correlation results will be unreliable.")

    # Create rankings
    comparison["imd_rank"] = comparison["imd_crime_score"].rank(ascending=False).astype(int)

    # Crime intensity rank from labels
    if intensity_col and intensity_col in comparison.columns:
        comparison["intensity_numeric"] = comparison[intensity_col].map(CRIME_INTENSITY_ORDER)
        # Handle unmapped values
        if comparison["intensity_numeric"].isna().any():
            # Try to infer from unique values
            unique_labels = comparison[intensity_col].unique()
            print(f"Warning: Some labels not in CRIME_INTENSITY_ORDER: {unique_labels}")
            # Assign numeric values based on assumed ordering
            comparison["intensity_numeric"] = comparison["intensity_numeric"].fillna(2)
        comparison["intensity_rank"] = comparison["intensity_numeric"].rank(
            ascending=False, method="min"
        ).astype(int)
    elif crimes_col and crimes_col in comparison.columns:
        # Use raw crime counts for ranking
        comparison["intensity_rank"] = comparison[crimes_col].rank(
            ascending=False
        ).astype(int)

    if crimes_col and crimes_col in comparison.columns:
        comparison["avg_monthly_crimes"] = comparison[crimes_col]

    # Compute correlations
    correlations = compute_correlations(comparison)

    # Display results
    print(f"\n{'='*60}")
    print("CORRELATION RESULTS")
    print(f"{'='*60}")

    if "spearman_r" in correlations:
        r = correlations["spearman_r"]
        p = correlations["spearman_p"]
        sig = "significant" if p < 0.05 else "not significant"
        strength = (
            "strong" if abs(r) > 0.7 else
            "moderate" if abs(r) > 0.4 else
            "weak"
        )
        print(f"\nSpearman rank correlation (intensity vs IMD):")
        print(f"  ρ = {r:.4f} ({strength} {'positive' if r > 0 else 'negative'})")
        print(f"  p = {p:.4f} ({sig} at α=0.05)")

    if "kendall_tau" in correlations:
        print(f"\nKendall's tau (robustness check):")
        print(f"  τ = {correlations['kendall_tau']:.4f}")
        print(f"  p = {correlations['kendall_p']:.4f}")

    if "spearman_crimes_vs_imd" in correlations:
        print(f"\nSpearman (raw crime count vs IMD crime score):")
        print(f"  ρ = {correlations['spearman_crimes_vs_imd']:.4f}")
        print(f"  p = {correlations['spearman_crimes_vs_imd_p']:.4f}")

    # Display comparison table
    display_name = name_col if name_col and name_col in comparison.columns else la_col
    display_cols = [c for c in [
        display_name, intensity_col, "intensity_rank",
        "imd_crime_score", "imd_rank"
    ] if c and c in comparison.columns]

    if display_cols:
        print(f"\nComparison Table (sorted by IMD rank):")
        print(comparison[display_cols].sort_values("imd_rank").to_string(index=False))

    # Rank agreement analysis
    if "intensity_rank" in comparison.columns and "imd_rank" in comparison.columns:
        comparison["rank_diff"] = abs(comparison["intensity_rank"] - comparison["imd_rank"])
        comparison["agreement"] = comparison["rank_diff"].apply(
            lambda d: "Strong" if d <= 2 else "Moderate" if d <= 4 else "Weak"
        )
        print(f"\nRank Agreement:")
        print(comparison["agreement"].value_counts().to_string())
        print(f"Mean rank difference: {comparison['rank_diff'].mean():.1f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Correlation results
    corr_df = pd.DataFrame([correlations])
    corr_path = OUTPUT_DIR / "imd_validation.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"\nSaved correlation results to {corr_path}")

    # Comparison table
    comp_path = OUTPUT_DIR / "imd_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"Saved comparison table to {comp_path}")

    # Interpretation guidance
    print(f"\n{'='*60}")
    print("INTERPRETATION FOR YOUR REPORT:")
    print(f"{'='*60}")
    if "spearman_r" in correlations:
        r = correlations["spearman_r"]
        p = correlations["spearman_p"]
        if r > 0.6 and p < 0.05:
            print("STRONG VALIDATION: Your crime intensity rankings align well")
            print("with the established IMD crime domain. This suggests your")
            print("rule-based system captures genuine patterns in recorded crime.")
        elif r > 0.3 and p < 0.1:
            print("MODERATE VALIDATION: Some alignment with IMD, suggesting")
            print("your system captures real patterns, though differences exist.")
            print("Discuss why: different time periods (IMD=2019, yours=2024),")
            print("different data sources, different aggregation methods.")
        else:
            print("WEAK/NO CORRELATION: This doesn't mean your system is wrong.")
            print("Discuss: IMD uses 2019 data vs your 2024 data; IMD includes")
            print("victim surveys while Police API is recorded crime only;")
            print("crime patterns may have shifted post-COVID.")

    print("\nCRITICAL: If using fallback IMD values, you MUST verify them")
    print("against the actual IoD2019 dataset before including in your report.")
    print("Download from: gov.uk/government/statistics/english-indices-of-deprivation-2019")
    print("\nDone.")


if __name__ == "__main__":
    main()
