"""Scope tests: every neighbourhood must belong to the council it is labelled with.

The neighbourhood list comes from police forces, which cover far more than one council
(Devon and Cornwall Police covers both Exeter and Cornwall, and everything in between).
These tests check the committed neighbourhood tables against the council boundaries.
"""
import json

import pandas as pd
import pytest
from shapely.geometry import Point, shape

from settings import BOUNDARIES_FILE, LA_CODES, PROCESSED_DIR

NEIGHBOURHOOD_TABLES = [
    "neighbourhood_features.parquet",
    "neighbourhood_classified.parquet",
    "neighbourhood_ml_classified.parquet",
]
# Derived tables without centroids: they must list exactly the same neighbourhoods.
DERIVED_TABLES = [
    "neighbourhood_anomalies.parquet",
    "neighbourhood_clusters.parquet",
]


@pytest.fixture(scope="module")
def councils():
    with open(BOUNDARIES_FILE, encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    return {f["properties"]["LAD24CD"]: shape(f["geometry"]) for f in features}


def test_boundary_file_has_all_study_councils(councils):
    assert set(councils) == set(LA_CODES)


@pytest.mark.parametrize("table", NEIGHBOURHOOD_TABLES)
def test_every_centroid_is_inside_its_labelled_council(table, councils):
    df = pd.read_parquet(PROCESSED_DIR / table)
    inside = [
        councils[code].covers(Point(lon, lat))
        for code, lon, lat in zip(df["la_code"], df["centroid_lon"], df["centroid_lat"])
    ]
    outside = df.loc[[not ok for ok in inside]]
    per_council = outside.groupby("la_name").size().to_dict()
    assert outside.empty, (
        f"{len(outside)} of {len(df)} neighbourhoods in {table} lie outside "
        f"the council they are labelled with: {per_council}"
    )


@pytest.mark.parametrize("table", NEIGHBOURHOOD_TABLES)
def test_no_neighbourhood_is_counted_under_two_councils(table):
    df = pd.read_parquet(PROCESSED_DIR / table)
    # The same police neighbourhood has the same id and the same centroid whichever
    # council it was downloaded for.
    key = ["neighbourhood_id", "centroid_lat", "centroid_lon"]
    duplicated = df[df.duplicated(key, keep=False)]
    pairs = sorted(duplicated["la_name"].unique())
    assert duplicated.empty, (
        f"{len(duplicated)} rows in {table} are the same neighbourhood listed under "
        f"more than one council: {pairs}"
    )


@pytest.mark.parametrize("table", DERIVED_TABLES)
def test_derived_tables_list_the_same_neighbourhoods(table):
    key = ["la_code", "neighbourhood_id"]
    base = pd.read_parquet(PROCESSED_DIR / "neighbourhood_features.parquet")[key]
    derived = pd.read_parquet(PROCESSED_DIR / table)[key]
    assert not derived.duplicated().any()
    assert set(map(tuple, derived.values)) == set(map(tuple, base.values))


def test_anomaly_summary_has_no_duplicated_council_rows():
    summary = pd.read_csv(PROCESSED_DIR / "anomaly_summary.csv")
    features = pd.read_parquet(PROCESSED_DIR / "neighbourhood_features.parquet")
    assert summary["la_name"].is_unique
    counts = features.groupby("la_name").size()
    assert dict(zip(summary["la_name"], summary["total_neighbourhoods"])) == counts.to_dict()
    assert int(summary["total_neighbourhoods"].sum()) == len(features)
