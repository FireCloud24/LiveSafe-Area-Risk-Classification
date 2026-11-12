"""How much of each council's recorded crime do its neighbourhoods account for?

Neighbourhood crime comes from the police neighbourhood boundaries; council crime comes from
a query on the (simplified) council polygon. The two need not match: a neighbourhood is
assigned whole to the council that contains its centroid, so it can bring in crime from
across the border (coverage above 100%) or leave part of the council out (below 100%).

Writes data/processed/neighbourhood_coverage.csv.
"""
from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).parent / "data" / "processed"


def coverage_table(neighbourhoods: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    per_council = neighbourhoods.groupby("la_name").agg(
        neighbourhoods=("avg_monthly_crimes", "size"),
        neighbourhood_avg_monthly_crimes=("avg_monthly_crimes", "sum"))
    out = per_council.join(
        areas.set_index("la_name")["avg_monthly_crimes"].rename("council_avg_monthly_crimes"), how="outer")
    out["coverage_pct"] = (out["neighbourhood_avg_monthly_crimes"] / out["council_avg_monthly_crimes"] * 100).round(1)
    out["neighbourhood_avg_monthly_crimes"] = out["neighbourhood_avg_monthly_crimes"].round(1)
    out["council_avg_monthly_crimes"] = out["council_avg_monthly_crimes"].round(1)
    return out.sort_values("coverage_pct").reset_index()


if __name__ == "__main__":
    table = coverage_table(pd.read_parquet(PROCESSED / "neighbourhood_features.parquet"),
                           pd.read_parquet(PROCESSED / "area_features.parquet"))
    table.to_csv(PROCESSED / "neighbourhood_coverage.csv", index=False)
    print(table.to_string(index=False))
