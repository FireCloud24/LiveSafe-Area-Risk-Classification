import json, time, urllib.request, urllib.parse, csv, sys
from pathlib import Path
from settings import LA_CODES, LA_NAME_MAP, START_MONTH, END_MONTH

# Output paths
ROOT            = Path(__file__).resolve().parent
NEIGH_DIR       = ROOT / "data" / "raw" / "neighbourhoods"
NEIGH_META_FILE = NEIGH_DIR / "neighbourhood_meta.json"
NEIGH_CRIME_FILE= NEIGH_DIR / "neighbourhood_crimes.csv"

NEIGH_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://data.police.uk/api"
SLEEP_SHORT = 0.5
SLEEP_LONG  = 1.0

# Force mapping: LA code → police force slug
LA_TO_FORCE = {
    "E08000019": "south-yorkshire",       # Sheffield
    "E08000035": "west-yorkshire",        # Leeds
    "E06000023": "avon-and-somerset",     # Bristol
    "E08000026": "west-midlands",         # Coventry
    "E08000021": "northumbria",           # Newcastle
    "E06000016": "leicestershire",        # Leicester
    "E07000148": "norfolk",               # Norwich
    "E07000041": "devon-and-cornwall",    # Exeter
    "E06000047": "durham",                # County Durham
    "E06000054": "wiltshire",             # Wiltshire
    "E06000019": "west-mercia",           # Herefordshire
    "E06000065": "north-yorkshire",       # North Yorkshire
    "E06000052": "devon-and-cornwall",    # Cornwall
    "W06000015": "south-wales",           # Cardiff
    "W06000011": "south-wales",           # Swansea
}


def api_get(url: str, retries: int = 3) -> list | dict | None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"      Rate limited — waiting 10s...")
                time.sleep(10)
            else:
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
    return None


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


def get_neighbourhoods(force: str) -> list[dict]:
    url  = f"{BASE}/{force}/neighbourhoods"
    data = api_get(url)
    return data if isinstance(data, list) else []


def get_neighbourhood_boundary(force: str, neighbourhood_id: str) -> list | None:
    url  = f"{BASE}/{force}/{urllib.parse.quote(neighbourhood_id)}/boundary"
    data = api_get(url)
    return data if isinstance(data, list) else None


def boundary_to_poly_param(boundary: list) -> str:
    step    = max(1, len(boundary) // 50)
    sampled = boundary[::step]
    return ":".join(f"{p['latitude']},{p['longitude']}" for p in sampled)


def bounding_box_from_boundary(boundary: list) -> str:
    lats = [float(p["latitude"])  for p in boundary]
    lons = [float(p["longitude"]) for p in boundary]
    corners = [
        (min(lats), min(lons)), (min(lats), max(lons)),
        (max(lats), max(lons)), (max(lats), min(lons)),
    ]
    return ":".join(f"{lat},{lon}" for lat, lon in corners)


def fetch_crimes_for_poly(poly_param: str, month: str) -> list:
    params = urllib.parse.urlencode({"poly": poly_param, "date": month})
    url    = f"{BASE}/crimes-street/all-crime?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
            return data if isinstance(data, list) else []
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return None
        return []
    except Exception:
        return []


def download_neighbourhoods():
    months = month_range(START_MONTH, END_MONTH)
    print(f"Downloading neighbourhood data for {len(LA_CODES)} areas")
    print(f"Period: {START_MONTH} to {END_MONTH} ({len(months)} months)\n")

    all_meta   = []
    total_rows = 0

    with open(NEIGH_CRIME_FILE, "w", newline="", encoding="utf-8") as crime_fh:
        writer = csv.DictWriter(crime_fh, fieldnames=[
            "la_code", "la_name", "force", "neighbourhood_id",
            "neighbourhood_name", "month", "category", "outcome_category",
        ])
        writer.writeheader()

        for la_code in LA_CODES:
            la_name = LA_NAME_MAP[la_code]
            force   = LA_TO_FORCE.get(la_code)

            if not force:
                print(f"  SKIP {la_name} — no force mapping")
                continue

            print(f"\n{'='*55}")
            print(f"  {la_name} ({la_code}) → force: {force}")
            print(f"{'='*55}")

            #  Get neighbourhood list
            neighbourhoods = get_neighbourhoods(force)
            if not neighbourhoods:
                print(f"  WARNING: no neighbourhoods returned for {force}")
                continue

            print(f"  {len(neighbourhoods)} neighbourhoods found")
            time.sleep(SLEEP_SHORT)

            for i, neigh in enumerate(neighbourhoods):
                nid   = neigh.get("id","")
                nname = neigh.get("name","")

                if not nid:
                    continue

                print(f"  [{i+1}/{len(neighbourhoods)}] {nname}")

                # Get boundary
                boundary = get_neighbourhood_boundary(force, nid)
                time.sleep(SLEEP_SHORT)

                if not boundary:
                    print(f"    No boundary — skipping")
                    continue

                # Store metadata
                lats = [float(p["latitude"])  for p in boundary]
                lons = [float(p["longitude"]) for p in boundary]
                all_meta.append({
                    "la_code":            la_code,
                    "la_name":            la_name,
                    "force":              force,
                    "neighbourhood_id":   nid,
                    "neighbourhood_name": nname,
                    "centroid_lat":       sum(lats)/len(lats),
                    "centroid_lon":       sum(lons)/len(lons),
                    "boundary_points":    len(boundary),
                })

                # Download crimes for each month
                poly    = boundary_to_poly_param(boundary)
                bbox    = bounding_box_from_boundary(boundary)
                n_crimes= 0

                for month in months:
                    crimes = fetch_crimes_for_poly(poly, month)

                    # Fallback to bounding box if polygon rejected
                    if crimes is None:
                        crimes = fetch_crimes_for_poly(bbox, month)
                        if crimes is None:
                            crimes = []

                    for c in crimes:
                        outcome = c.get("outcome_status") or {}
                        writer.writerow({
                            "la_code":            la_code,
                            "la_name":            la_name,
                            "force":              force,
                            "neighbourhood_id":   nid,
                            "neighbourhood_name": nname,
                            "month":              month,
                            "category":           c.get("category",""),
                            "outcome_category":   outcome.get("category",""),
                        })
                        n_crimes  += 1
                        total_rows += 1

                    time.sleep(SLEEP_SHORT)

                print(f"    {n_crimes} crimes across {len(months)} months")
                time.sleep(SLEEP_SHORT)

    # Save metadata
    with open(NEIGH_META_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_meta, fh, indent=2)

    print(f"\n{'='*55}")
    print(f"DONE")
    print(f"  {len(all_meta)} neighbourhoods processed")
    print(f"  {total_rows:,} crime records saved")
    print(f"  Meta: {NEIGH_META_FILE}")
    print(f"  Crimes: {NEIGH_CRIME_FILE}")


if __name__ == "__main__":
    download_neighbourhoods()
