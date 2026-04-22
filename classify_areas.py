import sys
import pandas as pd
import numpy as np
from settings import (
    PROCESSED_DIR,
    CRIME_INTENSITY_PERCENTILES,
    WEATHER_PRECIP_PERCENTILES,
    CRIME_TREND_STABLE_RANGE,
)

INPUT_FILE  = PROCESSED_DIR / "area_features.parquet"
OUTPUT_FILE = PROCESSED_DIR / "area_classified.parquet"


def classify_crime_intensity(series: pd.Series) -> pd.Series:
    p25, p50, p75 = [series.quantile(p / 100) for p in CRIME_INTENSITY_PERCENTILES]
    def label(v):
        if pd.isna(v):   return "Unknown"
        if v <= p25:     return "Very Low Recorded Crime"
        if v <= p50:     return "Moderate Crime Level"
        if v <= p75:     return "Elevated Crime Level"
        return                  "Severe Crime Concentration"
    return series.map(label)


def classify_crime_pattern(df: pd.DataFrame) -> pd.Series:
    def label(row):
        v, p = row["avg_violent_share"], row["avg_property_share"]
        if pd.isna(v) or pd.isna(p):   return "Unknown"
        if v >= 0.30 and v > p:         return "Violence-Dominant"
        if p >= 0.30 and p > v:         return "Property-Crime-Dominant"
        if v >= 0.20 and p >= 0.20:     return "Mixed Crime Pattern"
        return "Low-Complexity Crime Pattern"
    return df.apply(label, axis=1)


def classify_crime_trend(df: pd.DataFrame) -> pd.Series:
    lo, hi        = CRIME_TREND_STABLE_RANGE
    var_p75       = df["crime_variability"].quantile(0.75)
    def label(row):
        slope = row["crime_trend_slope"]
        var   = row["crime_variability"]
        if pd.isna(slope):      return "Insufficient Data"
        if var > var_p75:       return "Fluctuating"
        if slope < lo:          return "Improving"
        if slope > hi:          return "Deteriorating"
        return "Stable"
    return df.apply(label, axis=1)


def classify_weather_exposure(df: pd.DataFrame) -> pd.Series:
    p33, p66  = [df["avg_monthly_prcp"].quantile(p / 100) for p in WEATHER_PRECIP_PERCENTILES]
    pvar_p75  = df["prcp_variability"].quantile(0.75)
    def label(row):
        prcp = row["avg_monthly_prcp"]
        cold = row["cold_month_count"]
        pvar = row["prcp_variability"]
        if pd.isna(prcp):                          return "Unknown"
        if cold >= 4 and prcp >= p66:              return "High Weather Stress"
        if pvar > pvar_p75:                        return "Seasonally Volatile"
        if cold >= 4:                              return "Cold-Exposed"
        if prcp >= p66:                            return "Rain-Exposed"
        return "Mild Conditions"
    return df.apply(label, axis=1)


def derive_overall_profile(df: pd.DataFrame) -> pd.Series:
    def label(row):
        ci  = row["crime_intensity_label"]
        ct  = row["crime_trend_label"]
        we  = row["weather_exposure_label"]
        high   = ci in ("Elevated Crime Level", "Severe Crime Concentration")
        low    = ci in ("Very Low Recorded Crime", "Moderate Crime Level")
        badwx  = we in ("High Weather Stress", "Rain-Exposed", "Cold-Exposed")
        if high and badwx:          return "Persistent Multi-Factor Risk Area"
        if high and ct == "Deteriorating": return "Emerging Risk Area"
        if high and ct == "Fluctuating":   return "Volatile Risk Area"
        if high:                    return "Crime-Sensitive Area"
        if badwx and low:           return "Weather-Sensitive Area"
        if badwx:                   return "Structurally Exposed Area"
        if low and ct in ("Stable", "Improving"): return "Stable Low-Risk Area"
        return "Mixed-Risk Area"
    return df.apply(label, axis=1)


def classify_areas():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  {len(df)} areas")

    df["crime_intensity_label"]  = classify_crime_intensity(df["avg_monthly_crimes"])
    df["crime_pattern_label"]    = classify_crime_pattern(df)
    df["crime_trend_label"]      = classify_crime_trend(df)
    df["weather_exposure_label"] = classify_weather_exposure(df)
    df["overall_profile"]        = derive_overall_profile(df)

    cols = ["la_name", "overall_profile", "crime_intensity_label",
            "crime_pattern_label", "crime_trend_label", "weather_exposure_label"]
    print("\nClassification results:")
    print(df[cols].to_string(index=False))

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    classify_areas()
