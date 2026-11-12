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
