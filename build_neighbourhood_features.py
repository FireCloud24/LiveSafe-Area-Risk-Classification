import json, sys
import pandas as pd
import numpy as np
from pathlib import Path
from settings import PROCESSED_DIR, VIOLENT_CATEGORIES, PROPERTY_CATEGORIES

ROOT        = Path(__file__).resolve().parent
NEIGH_DIR   = ROOT / "data" / "raw" / "neighbourhoods"
META_FILE   = NEIGH_DIR / "neighbourhood_meta.json"
CRIMES_FILE = NEIGH_DIR / "neighbourhood_crimes.csv"
OUT_FEAT    = PROCESSED_DIR / "neighbourhood_features.parquet"
OUT_CLASS   = PROCESSED_DIR / "neighbourhood_classified.parquet"


def build_neighbourhood_features():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not META_FILE.exists() or not CRIMES_FILE.exists():
        print("ERROR: Run download_neighbourhoods.py first.")
        sys.exit(1)

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

    feat.to_parquet(OUT_FEAT, index=False)
    print(f"  Features saved: {len(feat)} neighbourhoods")

    # Classify neighbourhoods within each LA
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

    classified = pd.concat(classified_frames, ignore_index=True)
    classified.to_parquet(OUT_CLASS, index=False)
    print(f"  Classified: {len(classified)} neighbourhoods → {OUT_CLASS.name}")

    # Summary
    print("\nNeighbourhood counts per LA:")
    summary = classified.groupby("la_name")["neighbourhood_id"].count().sort_values(ascending=False)
    print(summary.to_string())


if __name__ == "__main__":
    build_neighbourhood_features()
