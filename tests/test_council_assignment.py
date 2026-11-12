"""Unit tests for council_assignment on a toy map: two square councils side by side."""
import pandas as pd
from shapely.geometry import box

import council_assignment
from council_assignment import assign_councils, council_of_point, scope_summary

COUNCILS = {"A": box(0, 0, 1, 1), "B": box(2, 0, 3, 1)}  # (lon, lat) squares


def rows(*items):
    return pd.DataFrame(
        [{"la_code": code, "la_name": code, "neighbourhood_id": nid,
          "centroid_lat": lat, "centroid_lon": lon, "total_crimes": 10}
         for code, nid, lat, lon in items]
    )


def test_council_of_point():
    assert council_of_point(0.5, 0.5, COUNCILS) == "A"
    assert council_of_point(0.5, 2.5, COUNCILS) == "B"
    assert council_of_point(0.5, 1.5, COUNCILS) is None  # in the gap between them


def test_assign_relabels_drops_and_deduplicates(monkeypatch):
    monkeypatch.setattr(council_assignment, "LA_NAME_MAP", {"A": "Alpha", "B": "Beta"})
    df = rows(
        ("A", "n1", 0.5, 0.5),   # correct
        ("A", "n2", 0.5, 2.5),   # labelled A, really in B
        ("B", "n2", 0.5, 2.5),   # the same neighbourhood again, labelled B
        ("A", "n3", 0.5, 1.5),   # outside both councils
        ("B", "n1", 0.2, 2.2),   # a different neighbourhood that happens to share an id
    )
    fixed, report = assign_councils(df, COUNCILS)

    assert sorted(zip(fixed["la_code"], fixed["neighbourhood_id"])) == [
        ("A", "n1"), ("B", "n1"), ("B", "n2")]
    assert set(fixed["la_name"]) == {"Alpha", "Beta"}
    assert list(report["rows_before"]) == [3, 2]
    assert list(report["rows_after"]) == [1, 2]
    assert list(report["label_correct_before"]) == [1, 2]

    summary = scope_summary(df, COUNCILS)
    assert summary == {
        "rows": 5, "inside_labelled_council": 3, "inside_another_study_council": 1,
        "outside_all_study_councils": 1, "duplicate_rows": 1, "unique_neighbourhoods": 4,
    }


def test_assign_is_idempotent(monkeypatch):
    monkeypatch.setattr(council_assignment, "LA_NAME_MAP", {"A": "Alpha", "B": "Beta"})
    df = rows(("A", "n1", 0.5, 0.5), ("A", "n2", 0.5, 2.5), ("A", "n3", 0.5, 1.5))
    once, _ = assign_councils(df, COUNCILS)
    twice, _ = assign_councils(once, COUNCILS)
    pd.testing.assert_frame_equal(once, twice)
