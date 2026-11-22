
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hackathon_preli_dataset.xlsx"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "weekly_features.csv"

REQUIRED_COLUMNS = [
    "date",
    "city",
    "ward_id",
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "dengue_cases",
]


def load_data():
    """Load and validate the raw Excel dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_excel(RAW_DATA_PATH)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError("Invalid or missing dates found.")

    
    numeric_columns = [
        "population_density_sqkm",
        "water_logging_risk_index",
        "sanitation_score",
        "avg_temp_c",
        "max_temp_c",
        "min_temp_c",
        "rainfall_mm",
        "relative_humidity_pct",
        "dengue_cases",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    
    missing_counts = df[REQUIRED_COLUMNS].isna().sum()

    if missing_counts.any():
        raise ValueError(
            "Missing values detected:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    
    duplicate_count = df.duplicated(
        subset=["ward_id", "date"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate ward-date records."
        )

    
    df = df.sort_values(
        ["ward_id", "date"]
    ).reset_index(drop=True)

    return df


def create_weather_features(df):
    """Create ward-level multi-week weather lag features."""

    result = df.copy()

    
    for week in range(1, 5):
        result[f"rainfall_lag_{week}w"] = (
            result.groupby("ward_id")["rainfall_mm"]
            .shift(week)
        )

    
    for window in [2, 3, 4]:
        result[f"rainfall_sum_{window}w"] = (
            result.groupby("ward_id")["rainfall_mm"]
            .transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window
                ).sum()
            )
        )

    
    for col in ["avg_temp_c", "relative_humidity_pct"]:
        result[f"{col}_avg_4w"] = (
            result.groupby("ward_id")[col]
            .transform(
                lambda series: series.rolling(
                    window=4,
                    min_periods=4
                ).mean()
            )
        )

    return result


def main():
    print("\n========== DENGUE WEATHER PIPELINE ==========\n")

    
    df = load_data()

    print("1. Raw dataset loaded successfully")
    print(f"Rows: {len(df):,}")
    print(f"Wards: {df['ward_id'].nunique()}")
    print(f"Cities: {df['city'].nunique()}")

    # Check weekly date gaps
    date_gaps = (
        df.groupby("ward_id")["date"]
        .diff()
        .dropna()
    )

    unexpected_gaps = (date_gaps != pd.Timedelta(days=7)).sum()

    print(f"Unexpected date gaps: {unexpected_gaps}")

    # Create features
    features_df = create_weather_features(df)

    print("\n2. Weather features created")

    feature_columns = [
        col for col in features_df.columns
        if "lag_" in col
        or "sum_" in col
        or "_avg_4w" in col
    ]

    for col in feature_columns:
        print(f"- {col}")

    
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n3. Processed dataset saved")
    print(f"File: {OUTPUT_PATH}")
    print(f"Final rows: {len(features_df):,}")
    print(f"Final columns: {len(features_df.columns)}")

    print("\n========== PIPELINE COMPLETED ==========\n")


if __name__ == "__main__":
    main()

