import sys
import pandas as pd
import numpy as np
from settings import CLEANED_DIR, PROCESSED_DIR, COLD_MONTH_TEMP_THRESHOLD, LA_NAME_MAP

CRIME_CLEAN     = CLEANED_DIR   / "crime_clean.parquet"
WEATHER_CLEAN   = CLEANED_DIR   / "weather_clean.parquet"
CRIME_MONTHLY   = PROCESSED_DIR / "crime_monthly.parquet"
WEATHER_MONTHLY = PROCESSED_DIR / "weather_monthly.parquet"
AREA_FEATURES   = PROCESSED_DIR / "area_features.parquet"


def build_crime_monthly(df: pd.DataFrame) -> pd.DataFrame:
    grp     = df.groupby(["la_code", "month"])
    monthly = grp.agg(
        total_crimes   = ("category",   "count"),
        violent_count  = ("is_violent",  "sum"),
        property_count = ("is_property", "sum"),
    ).reset_index()
    monthly["violent_share"]  = monthly["violent_count"]  / monthly["total_crimes"]
    monthly["property_share"] = monthly["property_count"] / monthly["total_crimes"]
    return monthly


def build_weather_monthly(df: pd.DataFrame) -> pd.DataFrame:
    grp     = df.groupby(["la_code", "month"])
    monthly = grp.agg(
        avg_tavg   = ("tavg", "mean"),
        min_tmin   = ("tmin", "min"),
        max_tmax   = ("tmax", "max"),
        total_prcp = ("prcp", "sum"),
        days       = ("date", "count"),
    ).reset_index()
    monthly["is_cold_month"] = (monthly["avg_tavg"] < COLD_MONTH_TEMP_THRESHOLD).astype(int)
    return monthly


def build_area_features(crime_m: pd.DataFrame, weather_m: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(crime_m, weather_m, on=["la_code", "month"], how="inner")

    def crime_slope(group):
        if len(group) < 3:
            return np.nan
        x = np.arange(len(group))
        y = group["total_crimes"].values
        return float(np.polyfit(x, y, 1)[0])

    trend = (
        merged.sort_values("month")
        .groupby("la_code")
        .apply(crime_slope)
        .reset_index()
        .rename(columns={0: "crime_trend_slope"})
    )

    area = merged.groupby("la_code").agg(
        avg_monthly_crimes = ("total_crimes",   "mean"),
        total_crimes_12m   = ("total_crimes",   "sum"),
        avg_violent_share  = ("violent_share",  "mean"),
        avg_property_share = ("property_share", "mean"),
        crime_variability  = ("total_crimes",   "std"),
        avg_tavg           = ("avg_tavg",        "mean"),
        avg_min_tmin       = ("min_tmin",        "mean"),
        avg_max_tmax       = ("max_tmax",        "mean"),
        avg_monthly_prcp   = ("total_prcp",      "mean"),
        total_prcp_12m     = ("total_prcp",      "sum"),
        cold_month_count   = ("is_cold_month",   "sum"),
        prcp_variability   = ("total_prcp",      "std"),
        months_with_data   = ("month",           "count"),
    ).reset_index()

    area = pd.merge(area, trend, on="la_code", how="left")
    area["la_name"] = area["la_code"].map(LA_NAME_MAP)
    front = ["la_code", "la_name", "months_with_data"]
    area  = area[front + [c for c in area.columns if c not in front]]
    return area


def build_features():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading cleaned data...")
    crime   = pd.read_parquet(CRIME_CLEAN)
    weather = pd.read_parquet(WEATHER_CLEAN)

    print("Building crime monthly aggregates...")
    crime_m = build_crime_monthly(crime)
    crime_m.to_parquet(CRIME_MONTHLY, index=False)
    print(f"  {len(crime_m)} area-month rows saved")

    print("Building weather monthly aggregates...")
    weather_m = build_weather_monthly(weather)
    weather_m.to_parquet(WEATHER_MONTHLY, index=False)
    print(f"  {len(weather_m)} area-month rows saved")

    print("Building area-level features...")
    area = build_area_features(crime_m, weather_m)
    area.to_parquet(AREA_FEATURES, index=False)
    print(f"  {len(area)} areas saved")

    print("\nFeature summary:")
    print(area[["la_name", "avg_monthly_crimes", "avg_violent_share",
                "avg_monthly_prcp", "cold_month_count", "crime_trend_slope"]].to_string(index=False))


if __name__ == "__main__":
    build_features()
