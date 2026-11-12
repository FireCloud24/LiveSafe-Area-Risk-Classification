"""Download a longer monthly crime series per council, for forecast evaluation.

    python download_crime_history.py

The main pipeline (crime.py) covers the 12 months of 2024, which is too short to compare a
forecast against a seasonal baseline: "the same month last year" needs more than a year.
This script downloads every month the police.uk API currently serves (about three years)
and stores only monthly totals per council in data/processed/crime_monthly_history.csv.

It uses the same 50-point polygon as crime.py for every month, and never falls back to a
coarser polygon or a bounding box: those cover a different area, which would put a false
jump in the series. Where the API refuses a month because the area holds more than 10,000
crimes, the same polygon is cut into pieces and the pieces are added up. A month that
still fails is left empty.

The run is resumable: rows already in the output file are skipped.
"""
import csv, json, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

from shapely.geometry import Polygon, box

from crime import MAX_POINTS, fetch_with_poly, largest_ring, ring_to_poly_param, simplify_ring
from settings import (
    BOUNDARIES_FILE, LA_CODES, LA_NAME_MAP, PROCESSED_DIR,
    PROPERTY_CATEGORIES, VIOLENT_CATEGORIES,
)

DATES_URL   = "https://data.police.uk/api/crimes-street-dates"
OUTPUT_FILE = PROCESSED_DIR / "crime_monthly_history.csv"
SLEEP       = 1.0
WORKERS     = 3   # councils downloaded at once; well inside the API limit of 15 requests a second
FIELDS      = ["la_code", "la_name", "month", "total_crimes", "violent_count",
               "property_count", "polygon_points", "status"]


def available_months() -> list[str]:
    with urllib.request.urlopen(DATES_URL, timeout=60) as resp:
        return sorted(item["date"] for item in json.loads(resp.read()))


def split_in_two(polygon):
    """Cut a polygon in half across the longer side of its bounding box."""
    min_x, min_y, max_x, max_y = polygon.bounds
    if max_x - min_x >= max_y - min_y:
        mid = (min_x + max_x) / 2
        halves = [box(min_x, min_y, mid, max_y), box(mid, min_y, max_x, max_y)]
    else:
        mid = (min_y + max_y) / 2
        halves = [box(min_x, min_y, max_x, mid), box(min_x, mid, max_x, max_y)]
    pieces = []
    for half in halves:
        part = polygon.intersection(half)
        geoms = getattr(part, "geoms", [part])
        pieces.extend(g for g in geoms if g.geom_type == "Polygon" and g.area > 0)
    return pieces


def fetch_polygon(polygon, month, depth=0):
    """Crimes inside a polygon for one month, or (None, pieces, code) on failure.

    The API refuses areas holding more than 10,000 crimes (HTTP 503). When that happens
    the polygon is cut in two and the halves are fetched and added together.
    """
    ring = list(polygon.exterior.coords)
    code = 0
    for _ in range(2):  # one retry
        crimes, code = fetch_with_poly(ring_to_poly_param(ring), month)
        if code == 200:
            return crimes, 1, 200
        time.sleep(SLEEP * 3)
    if code != 503 or depth >= 3:
        return None, 0, code
    crimes, pieces = [], 0
    for piece in split_in_two(polygon):
        part, n, code = fetch_polygon(piece, month, depth + 1)
        if part is None:
            return None, 0, code
        crimes.extend(part)
        pieces += n
        time.sleep(SLEEP)
    return crimes, pieces, 200


def fetch_month(ring, month):
    """Return (crimes, polygon_points, status) for the council's 50-point polygon."""
    simplified = simplify_ring(ring, MAX_POINTS)

    # First choice: exactly the request crime.py makes, so the totals match the main pipeline
    code = 0
    for _ in range(2):
        crimes, code = fetch_with_poly(ring_to_poly_param(simplified), month)
        if code == 200:
            return crimes, MAX_POINTS, "ok"
        time.sleep(SLEEP * 3)
    if code != 503:
        return None, None, f"failed (HTTP {code})"

    # Too many crimes for one request: cut the same polygon into pieces and add them up
    polygon = Polygon(simplified)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # repair self-crossings left by the simplification
        if polygon.geom_type != "Polygon":
            polygon = max(polygon.geoms, key=lambda g: g.area)
    crimes, pieces = [], 0
    for piece in split_in_two(polygon):
        part, n, code = fetch_polygon(piece, month, depth=1)
        if part is None:
            return None, None, f"failed (HTTP {code})"
        crimes.extend(part)
        pieces += n
        time.sleep(SLEEP)
    # A crime can sit exactly on a cut line and be returned by both halves
    unique = {c.get("persistent_id") or f"id-{c.get('id')}": c for c in crimes}
    return list(unique.values()), MAX_POINTS, f"ok (area split into {pieces} requests)"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(BOUNDARIES_FILE, encoding="utf-8") as fh:
        geometry = {f["properties"]["LAD24CD"]: f["geometry"] for f in json.load(fh)["features"]}

    done = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as fh:
            done = {(r["la_code"], r["month"]) for r in csv.DictReader(fh) if r["status"].startswith("ok")}

    months = available_months()
    print(f"API serves {len(months)} months: {months[0]} to {months[-1]}")

    rows = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["status"].startswith("ok")]

    lock = threading.Lock()

    def download_council(la_code):
        ring = largest_ring(geometry[la_code])
        la_name = LA_NAME_MAP[la_code]
        for month in months:
            if (la_code, month) in done:
                continue
            crimes, points, status = fetch_month(ring, month)
            row = {"la_code": la_code, "la_name": la_name, "month": month,
                   "total_crimes": "", "violent_count": "", "property_count": "",
                   "polygon_points": points or "", "status": status}
            if crimes is not None:
                cats = [(c.get("category") or "").strip().lower() for c in crimes]
                row["total_crimes"]   = len(cats)
                row["violent_count"]  = sum(c in VIOLENT_CATEGORIES for c in cats)
                row["property_count"] = sum(c in PROPERTY_CATEGORIES for c in cats)
            with lock:
                rows.append(row)
                print(f"  {la_name} {month}: {row['total_crimes']} ({status}, {points} points)", flush=True)
                # Rewrite after every request so an interrupted run loses nothing
                with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(sorted(rows, key=lambda r: (r["la_code"], r["month"])))
            time.sleep(SLEEP)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(download_council, LA_CODES))

    failed = [r for r in rows if not r["status"].startswith("ok")]
    print(f"\nDone. {len(rows) - len(failed)} council-months saved, {len(failed)} failed.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
