"""Build neighbourhood features and within-council labels.

    python build_neighbourhood_features.py

Uses the raw neighbourhood download if it is present (data/raw/neighbourhoods/, written by
download_neighbourhoods.py). If it is not, it re-derives the tables from the committed
neighbourhood_features.parquet, which holds the same per-neighbourhood aggregates.

Either way every neighbourhood is assigned to the council its centroid lies in
(council_assignment.py); neighbourhoods outside all study councils are dropped.
"""
import json, sys
import pandas as pd
import numpy as np
from pathlib import Path
from settings import PROCESSED_DIR, VIOLENT_CATEGORIES, PROPERTY_CATEGORIES
from council_assignment import assign_councils, scope_summary

ROOT        = Path(__file__).resolve().parent
NEIGH_DIR   = ROOT / "data" / "raw" / "neighbourhoods"
META_FILE   = NEIGH_DIR / "neighbourhood_meta.json"
CRIMES_FILE = NEIGH_DIR / "neighbourhood_crimes.csv"
OUT_FEAT    = PROCESSED_DIR / "neighbourhood_features.parquet"
OUT_CLASS   = PROCESSED_DIR / "neighbourhood_classified.parquet"
OUT_REPORT  = PROCESSED_DIR / "neighbourhood_scope_fix.csv"
OUT_SUMMARY = PROCESSED_DIR / "neighbourhood_scope_fix_summary.json"

FEATURE_COLUMNS = [
    "la_code", "neighbourhood_id", "total_crimes", "violent_count", "property_count",
    "months_present", "avg_monthly_crimes", "violent_share", "property_share",
    "trend_slope", "top_crime_category", "la_name", "neighbourhood_name",
    "centroid_lat", "centroid_lon",
]


def features_from_raw() -> pd.DataFrame:
    """Aggregate the raw download to one row per (downloaded council label, neighbourhood)."""
    print("Loading neighbourhood data...")
    with open(META_FILE) as fh:
        meta = pd.DataFrame(json.load(fh))

    crimes = pd.read_csv(CRIMES_FILE, dtype=str)
    print(f"  {len(meta)} neighbourhoods, {len(crimes):,} crime records")

    # Flag crime types
    cat = crimes["category"].str.strip().str.lower()
    crimes["is_violent"]  = cat.isin(VIOLENT_CATEGORIES).astype(int)
    crimes["is_property"] = cat.isin(PROPERTY_CATEGORIES).astype(int)

    # Aggregate per neighbourhood
    grp = crimes.groupby(["la_code", "neighbourhood_id"])
    agg = grp.agg(
        total_crimes   = ("category",    "count"),
        violent_count  = ("is_violent",  "sum"),
        property_count = ("is_property", "sum"),
        months_present = ("month",       "nunique"),
    ).reset_index()

    agg["avg_monthly_crimes"] = agg["total_crimes"] / agg["months_present"].clip(lower=1)
    agg["violent_share"]      = agg["violent_count"]  / agg["total_crimes"].clip(lower=1)
    agg["property_share"]     = agg["property_count"] / agg["total_crimes"].clip(lower=1)

    # Crime trend per neighbourhood
    monthly = crimes.groupby(
        ["la_code","neighbourhood_id","month"]
    )["category"].count().reset_index().rename(columns={"category":"monthly_count"})

    def slope(g):
        if len(g) < 3: return np.nan
        x = np.arange(len(g))
        return float(np.polyfit(x, g["monthly_count"].values, 1)[0])

    trends = (monthly.sort_values("month")
              .groupby(["la_code","neighbourhood_id"])
              .apply(slope)
              .reset_index()
              .rename(columns={0: "trend_slope"}))

    # Top crime category per neighbourhood
    top_cat = (crimes.groupby(["la_code","neighbourhood_id","category"])
               ["category"].count()
               .reset_index(name="cat_count")
               .sort_values("cat_count", ascending=False)
               .groupby(["la_code","neighbourhood_id"])
               .first()
               .reset_index()[["la_code","neighbourhood_id","category"]]
               .rename(columns={"category":"top_crime_category"}))

    # Merge everything
    feat = agg.merge(trends, on=["la_code","neighbourhood_id"], how="left")
    feat = feat.merge(top_cat, on=["la_code","neighbourhood_id"], how="left")
    feat = feat.merge(
        meta[["la_code","la_name","neighbourhood_id","neighbourhood_name",
              "centroid_lat","centroid_lon"]],
        on=["la_code","neighbourhood_id"], how="left"
    )

    return feat[FEATURE_COLUMNS]


def classify_neighbourhoods(feat: pd.DataFrame) -> pd.DataFrame:
    """Rank and label neighbourhoods within their council."""
    classified_frames = []

    for la_code, group in feat.groupby("la_code"):
        g = group.copy()

        # Rank by avg monthly crimes within this LA (1 = highest crime)
        g["crime_rank"]  = g["avg_monthly_crimes"].rank(ascending=False, method="min").astype(int)
        g["total_neighs"]= len(g)
        g["rank_pct"]    = g["crime_rank"] / g["total_neighs"]  # 0=highest, 1=lowest

        # Crime intensity label within LA
        p33 = g["avg_monthly_crimes"].quantile(0.33)
        p66 = g["avg_monthly_crimes"].quantile(0.66)

        def intensity(v):
            if v >= p66: return "High Crime"
            if v >= p33: return "Moderate Crime"
            return "Lower Crime"

        g["neigh_intensity"] = g["avg_monthly_crimes"].map(intensity)

        # Trend label
        def trend_label(s):
            if pd.isna(s):   return "Stable"
            if s > 2:        return "Worsening"
            if s < -2:       return "Improving"
            return "Stable"

        g["neigh_trend"] = g["trend_slope"].map(trend_label)

        classified_frames.append(g)

    return pd.concat(classified_frames, ignore_index=True)


def build_neighbourhood_features():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if META_FILE.exists() and CRIMES_FILE.exists():
        feat = features_from_raw()
    elif OUT_FEAT.exists():
        print(f"Raw download not found; re-deriving from {OUT_FEAT.name}")
        feat = pd.read_parquet(OUT_FEAT)[FEATURE_COLUMNS]
    else:
        print("ERROR: Run download_neighbourhoods.py first.")
        sys.exit(1)

    # Assign each neighbourhood to the council its centroid lies in
    before = scope_summary(feat)
    feat, report = assign_councils(feat)
    after = scope_summary(feat)

    print("\nCouncil assignment (by neighbourhood centroid):")
    print(f"  rows before                           : {before['rows']}")
    print(f"  inside the council they were labelled : {before['inside_labelled_council']}")
    print(f"  inside another study council          : {before['inside_another_study_council']}")
    print(f"  outside all study councils            : {before['outside_all_study_councils']}")
    print(f"  duplicate rows                        : {before['duplicate_rows']}")
    print(f"  rows after                            : {after['rows']}")
    print("\nNeighbourhoods per council, before and after:")
    print(report[["la_name", "rows_before", "rows_after"]].to_string(index=False))

    if before != after:
        # Only written when the run actually changed something, so the record of the
        # original problem survives later re-runs on already-corrected data.
        report.to_csv(OUT_REPORT, index=False)
        with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
            json.dump({"before": before, "after": after}, fh, indent=2)
        print(f"\n  Report saved: {OUT_REPORT.name}, {OUT_SUMMARY.name}")

    feat.to_parquet(OUT_FEAT, index=False)
    print(f"  Features saved: {len(feat)} neighbourhoods")

    classified = classify_neighbourhoods(feat)
    classified.to_parquet(OUT_CLASS, index=False)
    print(f"  Classified: {len(classified)} neighbourhoods -> {OUT_CLASS.name}")


if __name__ == "__main__":
    build_neighbourhood_features()
