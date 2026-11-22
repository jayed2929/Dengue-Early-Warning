
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "weekly_features.csv"
MODEL_PATH = ROOT / "models" / "dengue_model.joblib"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Recreate future target for comparison
grouped = df.groupby("ward_id")["dengue_cases"]

df["target"] = (
    grouped.shift(-2)
    + grouped.shift(-3)
    + grouped.shift(-4)
)

# Recreate seasonal features
month = df["date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)

# Load model
bundle = joblib.load(MODEL_PATH)
features = bundle["features"]

# Keep complete rows
df = df.dropna(subset=features + ["target"]).copy()

# Test period
test = df[
    (df["date"] >= "2025-01-06")
    & (df["date"] <= "2025-11-24")
].copy()

# Generate predictions
predictions = bundle["model"].predict(test[features])
test["prediction"] = np.maximum(predictions, 0)

test["month"] = test["date"].dt.month

# Monthly comparison
monthly = test.groupby("month").agg(
    observations=("target", "count"),
    actual_mean=("target", "mean"),
    actual_median=("target", "median"),
    predicted_mean=("prediction", "mean"),
    predicted_median=("prediction", "median"),
).round(2)

print("\n========== MONTHLY MODEL ANALYSIS ==========")
print(monthly.to_string())

print("\n========== DECEMBER LATEST PREDICTIONS ==========")

latest = (
    df.sort_values("date")
    .groupby("ward_id")
    .tail(1)
    .copy()
)

latest["prediction"] = np.maximum(
    bundle["model"].predict(latest[features]), 0
)

print(
    latest[
        ["city", "ward_id", "date", "prediction"]
    ]
    .sort_values("prediction", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n=============================================")
