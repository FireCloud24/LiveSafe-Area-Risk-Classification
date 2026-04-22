import sys, json, calendar, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path
from settings import (
    BOUNDARIES_FILE, RAW_WEATHER_DIR,
    LA_CODES, LA_NAME_MAP, START_MONTH, END_MONTH,
)

try:
    from meteostat import Point, Daily, Stations
    import pandas as pd
except ImportError:
    print("ERROR: Run  pip install meteostat pandas")
    sys.exit(1)

OUTPUT_FILE = RAW_WEATHER_DIR / "weather_raw.csv"


def centroid(geometry: dict) -> tuple:
    gtype = geometry["type"]
    if gtype == "Polygon":
        coords = geometry["coordinates"][0]
    elif gtype == "MultiPolygon":
        coords = []
        for poly in geometry["coordinates"]:
            coords.extend(poly[0])
    else:
        raise ValueError(f"Unsupported geometry: {gtype}")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def fetch_weather_for_location(lat, lon, start_dt, end_dt):
    # Attempt 1: standard Point lookup
    df = Daily(Point(lat, lon), start_dt, end_dt).fetch()
    if not df.empty:
        return df

    # Attempt 2: find nearby stations explicitly, try up to 5
    stations = Stations()
    stations = stations.nearby(lat, lon)
    stations = stations.inventory("daily", (start_dt, end_dt))
    nearby   = stations.fetch(5)

    for station_id in nearby.index:
        df = Daily(station_id, start_dt, end_dt).fetch()
        if not df.empty and len(df) > 30:
            print(f"      (used station {station_id})")
            return df

    return pd.DataFrame()


def download_weather():
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

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

    sy, sm = int(START_MONTH[:4]), int(START_MONTH[5:7])
    ey, em = int(END_MONTH[:4]),   int(END_MONTH[5:7])
    start_dt = datetime(sy, sm, 1)
    end_dt   = datetime(ey, em, calendar.monthrange(ey, em)[1])

    all_frames = []
    print(f"Downloading weather: {len(la_geometry)} areas, {START_MONTH} to {END_MONTH}\n")

    for la_code in LA_CODES:
        if la_code not in la_geometry:
            print(f"  SKIP {la_code} — no geometry")
            continue

        la_name  = LA_NAME_MAP[la_code]
        lat, lon = centroid(la_geometry[la_code])
        print(f"  {la_name} ({la_code})  centroid: {lat:.3f}, {lon:.3f}")

        df = fetch_weather_for_location(lat, lon, start_dt, end_dt)

        if df.empty:
            print(f"    WARNING: no weather data found — skipping")
            continue

        df = df.reset_index().rename(columns={"time": "date"})
        df["la_code"] = la_code
        df["la_name"] = la_name
        df["month"]   = df["date"].dt.strftime("%Y-%m")

        keep = ["la_code", "la_name", "date", "month", "tavg", "tmin", "tmax", "prcp"]
        df   = df[[c for c in keep if c in df.columns]]

        missing_prcp = int(df["prcp"].isna().sum()) if "prcp" in df.columns else "N/A"
        print(f"    {len(df)} days  |  prcp missing: {missing_prcp}")
        all_frames.append(df)

    if not all_frames:
        print("ERROR: no weather data collected at all.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(combined):,} daily records saved to {OUTPUT_FILE}")
    print(f"Areas with data: {combined['la_code'].nunique()} / {len(LA_CODES)}")


if __name__ == "__main__":
    download_weather()
