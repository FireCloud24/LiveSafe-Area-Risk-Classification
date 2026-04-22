import json, sys, time, csv, urllib.request, urllib.parse
from pathlib import Path
from typing import Any
from settings import (
    BOUNDARIES_FILE, RAW_CRIME_DIR,
    LA_CODES, LA_NAME_MAP, START_MONTH, END_MONTH,
)

POLICE_API   = "https://data.police.uk/api/crimes-street/all-crime"
OUTPUT_FILE  = RAW_CRIME_DIR / "crime_raw.csv"
SLEEP        = 1.0
MAX_POINTS   = 50


def month_range(start, end):
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]),   int(end[5:7])
    months, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def largest_ring(geometry):
    gtype = geometry["type"]
    if gtype == "Polygon":
        rings = [geometry["coordinates"][0]]
    elif gtype == "MultiPolygon":
        rings = [poly[0] for poly in geometry["coordinates"]]
    else:
        raise ValueError(f"Unsupported geometry: {gtype}")
    return max(rings, key=len)


def simplify_ring(ring, max_points):
    if len(ring) <= max_points:
        return ring
    step = len(ring) / max_points
    return [ring[int(i * step)] for i in range(max_points)]


def ring_to_poly_param(ring):
    return ":".join(f"{lat},{lon}" for lon, lat in ring)


def bounding_box_poly(ring):
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    corners = [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon),
    ]
    return ":".join(f"{lat},{lon}" for lat, lon in corners)


def fetch_with_poly(poly_param, month):
    params = urllib.parse.urlencode({"poly": poly_param, "date": month})
    url    = f"{POLICE_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.loads(resp.read()), 200
    except urllib.error.HTTPError as e:
        return [], e.code
    except Exception as e:
        print(f"      Network error: {e}")
        return [], 0


def fetch_crimes_robust(ring, month, la_name):
    # Strategy 1: 50-point polygon
    simplified = simplify_ring(ring, MAX_POINTS)
    crimes, code = fetch_with_poly(ring_to_poly_param(simplified), month)
    if code == 200:
        return crimes

    # Strategy 2: 20-point polygon
    tiny = simplify_ring(ring, 20)
    crimes, code = fetch_with_poly(ring_to_poly_param(tiny), month)
    if code == 200:
        return crimes

    # Strategy 3: bounding box
    crimes, code = fetch_with_poly(bounding_box_poly(ring), month)
    if code == 200:
        print(f"      (bounding box fallback used)")
        return crimes

    print(f"      All strategies failed for {la_name} {month}")
    return []


def download_crime():
    RAW_CRIME_DIR.mkdir(parents=True, exist_ok=True)

    if not BOUNDARIES_FILE.exists():
        print("ERROR: Run  python boundaries.py  first.")
        sys.exit(1)

    with open(BOUNDARIES_FILE, encoding="utf-8") as fh:
        geo = json.load(fh)

    la_geometry = {
        f["properties"]["LAD24CD"]: f["geometry"]
        for f in geo["features"]
        if f["properties"].get("LAD24CD") in set(LA_CODES)
    }

    months     = month_range(START_MONTH, END_MONTH)
    total_rows = 0
    fieldnames = [
        "la_code", "la_name", "month", "category",
        "street_id", "street_name", "latitude", "longitude",
        "outcome_category", "outcome_date",
    ]

    print(f"Downloading crime: {len(la_geometry)} areas x {len(months)} months")
    print(f"Using {MAX_POINTS}-point polygons with fallback strategies\n")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for la_code in LA_CODES:
            if la_code not in la_geometry:
                print(f"  SKIP {la_code} — no geometry")
                continue

            la_name = LA_NAME_MAP[la_code]
            ring    = largest_ring(la_geometry[la_code])
            print(f"\n  {la_name} ({la_code})  — ring: {len(ring)} points")

            for month in months:
                crimes = fetch_crimes_robust(ring, month, la_name)
                print(f"    {month}: {len(crimes)} crimes")

                for c in crimes:
                    loc     = c.get("location") or {}
                    street  = loc.get("street") or {}
                    outcome = c.get("outcome_status") or {}
                    writer.writerow({
                        "la_code":          la_code,
                        "la_name":          la_name,
                        "month":            month,
                        "category":         c.get("category", ""),
                        "street_id":        street.get("id", ""),
                        "street_name":      street.get("name", ""),
                        "latitude":         loc.get("latitude", ""),
                        "longitude":        loc.get("longitude", ""),
                        "outcome_category": outcome.get("category", ""),
                        "outcome_date":     outcome.get("date", ""),
                    })
                    total_rows += 1

                time.sleep(SLEEP)

    print(f"\nDone. {total_rows:,} records saved to {OUTPUT_FILE}")

    import pandas as pd
    df = pd.read_csv(OUTPUT_FILE)
    summary = df.groupby("la_name")["category"].count().sort_values(ascending=False)
    print("\nPer-area totals:")
    print(summary.to_string())
    zeros = summary[summary == 0]
    if not zeros.empty:
        print(f"\nWARNING: still 0 crimes for: {list(zeros.index)}")


if __name__ == "__main__":
    download_crime()
