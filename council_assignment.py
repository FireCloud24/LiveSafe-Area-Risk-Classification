"""Assign police neighbourhoods to the council they actually lie in.

Police neighbourhoods are listed per police force, and a force covers far more than one
council: Devon and Cornwall Police covers Exeter, Cornwall and everything between them.
Labelling every neighbourhood of a force with one study council therefore puts most
neighbourhoods in the wrong council, and counts some forces twice.

The fix here: take each neighbourhood's centroid, find which of the study councils'
boundary polygons contains it, relabel it with that council, and drop it if it is in none.
"""
from __future__ import annotations

import json

import pandas as pd
from shapely.geometry import Point, shape
from shapely.prepared import prep

from settings import BOUNDARIES_FILE, LA_NAME_MAP

# The same police neighbourhood has the same id and centroid whichever council it was
# downloaded for; ids alone are only unique within a force.
DEDUP_KEY = ["neighbourhood_id", "centroid_lat", "centroid_lon"]


def load_councils(path=BOUNDARIES_FILE) -> dict:
    """Council code -> boundary polygon, for the study councils."""
    with open(path, encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    return {f["properties"]["LAD24CD"]: shape(f["geometry"]) for f in features}


def council_of_point(lat: float, lon: float, councils: dict, prepared: dict | None = None):
    """Code of the study council containing the point, or None if it is in none of them."""
    prepared = prepared or {code: prep(geom) for code, geom in councils.items()}
    point = Point(lon, lat)
    for code, geom in prepared.items():
        if geom.covers(point):
            return code
    return None


def assign_councils(df: pd.DataFrame, councils: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel neighbourhoods by where their centroid lies.

    `df` needs la_code, la_name, neighbourhood_id, centroid_lat, centroid_lon.
    Returns (fixed, report). `fixed` has one row per neighbourhood that lies inside a study
    council, labelled with that council. `report` has one row per council with the counts
    before and after.
    """
    councils = councils or load_councils()
    prepared = {code: prep(geom) for code, geom in councils.items()}

    df = df.copy()
    df["_true_code"] = [
        council_of_point(lat, lon, councils, prepared)
        for lat, lon in zip(df["centroid_lat"], df["centroid_lon"])
    ]
    df["_label_was_right"] = df["_true_code"] == df["la_code"]

    inside = df[df["_true_code"].notna()]
    # Where a neighbourhood was downloaded under two councils, keep the correctly labelled copy.
    fixed = (inside.sort_values("_label_was_right", ascending=False, kind="stable")
             .drop_duplicates(DEDUP_KEY, keep="first")
             .copy())
    fixed["la_code"] = fixed["_true_code"]
    fixed["la_name"] = fixed["la_code"].map(LA_NAME_MAP)

    report = pd.DataFrame({
        "la_code": list(LA_NAME_MAP),
        "la_name": list(LA_NAME_MAP.values()),
    })
    counts = {
        "rows_before": df.groupby("la_code").size(),
        "label_correct_before": df[df["_label_was_right"]].groupby("la_code").size(),
        "rows_after": fixed.groupby("la_code").size(),
    }
    for name, series in counts.items():
        report[name] = report["la_code"].map(series).fillna(0).astype(int)
    report["rows_removed_or_moved"] = report["rows_before"] - report["label_correct_before"]

    fixed = (fixed.drop(columns=["_true_code", "_label_was_right"])
             .sort_values(["la_code", "neighbourhood_id"])
             .reset_index(drop=True))
    return fixed, report


def scope_summary(df: pd.DataFrame, councils: dict | None = None) -> dict:
    """Headline counts for a neighbourhood table: how many rows are where they claim to be."""
    councils = councils or load_councils()
    prepared = {code: prep(geom) for code, geom in councils.items()}
    true_code = pd.Series(
        [council_of_point(lat, lon, councils, prepared)
         for lat, lon in zip(df["centroid_lat"], df["centroid_lon"])],
        index=df.index,
    )
    return {
        "rows": int(len(df)),
        "inside_labelled_council": int((true_code == df["la_code"]).sum()),
        "inside_another_study_council": int((true_code.notna() & (true_code != df["la_code"])).sum()),
        "outside_all_study_councils": int(true_code.isna().sum()),
        "duplicate_rows": int(df.duplicated(DEDUP_KEY, keep="first").sum()),
        "unique_neighbourhoods": int(len(df.drop_duplicates(DEDUP_KEY))),
    }
