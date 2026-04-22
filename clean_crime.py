import sys
import pandas as pd
from pathlib import Path
from settings import RAW_CRIME_DIR, CLEANED_DIR, VIOLENT_CATEGORIES, PROPERTY_CATEGORIES

INPUT_FILE  = RAW_CRIME_DIR / "crime_raw.csv"
OUTPUT_FILE = CLEANED_DIR   / "crime_clean.parquet"


def clean_crime():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, dtype=str)
    print(f"  {len(df):,} rows loaded")

    before = len(df)
    df     = df.dropna(subset=["la_code", "month", "category"])
    print(f"  Dropped {before - len(df)} rows missing key fields")

    df["month"]       = df["month"].str.strip().str[:7]
    cat               = df["category"].str.strip().str.lower()
    df["is_violent"]  = cat.isin(VIOLENT_CATEGORIES).astype(int)
    df["is_property"] = cat.isin(PROPERTY_CATEGORIES).astype(int)

    for col in ["la_code", "la_name", "category", "street_name", "outcome_category"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    print(f"  {len(df):,} rows after cleaning")
    print(f"  Violent: {df['is_violent'].mean():.1%}  |  Property: {df['is_property'].mean():.1%}")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_crime()
