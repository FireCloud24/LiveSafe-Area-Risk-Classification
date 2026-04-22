import sys
import pandas as pd
from settings import RAW_WEATHER_DIR, CLEANED_DIR

INPUT_FILE  = RAW_WEATHER_DIR / "weather_raw.csv"
OUTPUT_FILE = CLEANED_DIR     / "weather_clean.parquet"
MISSING_THRESHOLD = 0.20


def clean_weather():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    print(f"  {len(df):,} rows loaded")

    before = len(df)
    df     = df.dropna(subset=["la_code", "date"])
    print(f"  Dropped {before - len(df)} rows missing la_code/date")

    if "month" not in df.columns:
        df["month"] = df["date"].dt.strftime("%Y-%m")

    df = df.sort_values(["la_code", "date"]).reset_index(drop=True)

    weather_cols = [c for c in ["tavg", "tmin", "tmax", "prcp"] if c in df.columns]
    df[weather_cols] = (
        df.groupby("la_code")[weather_cols]
        .transform(lambda g: g.ffill().bfill())
    )

    if "prcp" in df.columns:
        df["prcp"] = df["prcp"].fillna(0.0)

    for col in weather_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  {len(df):,} rows after cleaning")
    print(f"  Areas: {df['la_code'].nunique()}  |  {df['date'].min().date()} to {df['date'].max().date()}")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_weather()
