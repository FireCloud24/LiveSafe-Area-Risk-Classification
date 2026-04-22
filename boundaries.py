import json, sys, urllib.request
from pathlib import Path
from settings import EXTERNAL_DIR, LA_CODES

BOUNDARIES_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_December_2024_Boundaries_UK_BGC/FeatureServer/0/query"
    "?where=1%3D1&outFields=LAD24CD,LAD24NM&f=geojson&outSR=4326"
)
OUTPUT_FILE = EXTERNAL_DIR / "la_boundaries.geojson"


def download_boundaries():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading LA boundaries from ONS (30-60 seconds)...")
    try:
        with urllib.request.urlopen(BOUNDARIES_URL, timeout=120) as resp:
            raw = resp.read()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    data     = json.loads(raw)
    features = data.get("features", [])
    print(f"Downloaded {len(features)} total LA features.")

    filtered = [
        f for f in features
        if f.get("properties", {}).get("LAD24CD") in set(LA_CODES)
    ]
    print(f"Filtered to {len(filtered)} prototype areas.")

    if len(filtered) < len(LA_CODES):
        found   = {f["properties"]["LAD24CD"] for f in filtered}
        missing = set(LA_CODES) - found
        print(f"WARNING: codes not found in boundaries: {missing}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": filtered}, fh)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    download_boundaries()
