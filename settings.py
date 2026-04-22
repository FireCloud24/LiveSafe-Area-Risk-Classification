from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_CRIME_DIR    = ROOT / "data" / "raw" / "crime"
RAW_WEATHER_DIR  = ROOT / "data" / "raw" / "weather"
EXTERNAL_DIR     = ROOT / "data" / "external"
CLEANED_DIR      = ROOT / "data" / "cleaned"
PROCESSED_DIR    = ROOT / "data" / "processed"
BOUNDARIES_FILE  = EXTERNAL_DIR / "la_boundaries.geojson"
TEMPLATES_DIR    = ROOT / "templates"
STATIC_DIR       = ROOT / "static"

START_MONTH = "2024-01"
END_MONTH   = "2024-12"

LOCAL_AUTHORITIES = [
    ("E08000019", "Sheffield"),
    ("E08000035", "Leeds"),
    ("E06000023", "Bristol, City of"),
    ("E08000026", "Coventry"),
    ("E08000021", "Newcastle upon Tyne"),
    ("E06000016", "Leicester"),
    ("E07000148", "Norwich"),
    ("E07000041", "Exeter"),
    ("E06000047", "County Durham"),
    ("E06000054", "Wiltshire"),
    ("E06000019", "Herefordshire, County of"),
    ("E06000065", "North Yorkshire"),
    ("E06000052", "Cornwall"),
    ("W06000015", "Cardiff"),
    ("W06000011", "Swansea"),
]

LA_CODES    = [code for code, _ in LOCAL_AUTHORITIES]
LA_NAME_MAP = {code: name for code, name in LOCAL_AUTHORITIES}

VIOLENT_CATEGORIES = {
    "violent-crime",
    "robbery",
    "possession-of-weapons",
    "public-order",
}

PROPERTY_CATEGORIES = {
    "burglary",
    "theft-from-the-person",
    "shoplifting",
    "vehicle-crime",
    "bicycle-theft",
    "other-theft",
}

COLD_MONTH_TEMP_THRESHOLD   = 5.0
CRIME_INTENSITY_PERCENTILES = [25, 50, 75]
WEATHER_PRECIP_PERCENTILES  = [33, 66]
CRIME_TREND_STABLE_RANGE    = (-0.5, 0.5)

if __name__ == "__main__":
    print(f"ROOT         : {ROOT}")
    print(f"RAW_CRIME_DIR: {RAW_CRIME_DIR}")
    for code, name in LOCAL_AUTHORITIES:
        print(f"  {code}  {name}")
