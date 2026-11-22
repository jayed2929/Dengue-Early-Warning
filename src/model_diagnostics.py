
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "weekly_features.csv"
MODEL_PATH = ROOT / "models" / "dengue_model.joblib"


df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["ward_id", "date"]).copy()

grouped = df.groupby("ward_id")["dengue_cases"]

df["target"] = (
    grouped.shift(-2)
    + grouped.shift(-3)
    + grouped.shift(-4)
)

bundle = joblib.load(MODEL_PATH)

features = bundle["features"]

df["month_sin"] = np.sin(
    2 * np.pi * df["date"].dt.month / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["date"].dt.month / 12
)

df = df.dropna(subset=features + ["target"]).copy()

test = df[
    (df["date"] >= "2025-01-06")
    & (df["date"] <= "2025-11-24")
].copy()

predictions = bundle["model"].predict(test[features])

test["prediction"] = np.maximum(predictions, 0)

test["absolute_error"] = (
    test["target"] - test["prediction"]
).abs()

print("\n========== MODEL DIAGNOSTICS ==========")

print("\nTarget distribution:")
print(test["target"].describe())

print("\nPrediction distribution:")
print(test["prediction"].describe())

print("\nError by actual case group:")

test["case_group"] = pd.cut(
    test["target"],
    bins=[-0.001, 1, 5, 15, np.inf],
    labels=["0-1", "2-5", "6-15", "Above 15"]
)

print(
    test.groupby(
        "case_group",
        observed=False
    )["absolute_error"].agg(
        ["count", "mean", "median"]
    )
)

print("\nTarget completeness:")
print("Test rows:", len(test))
print("Missing targets:", test["target"].isna().sum())

print("\nOverall MAE:", mean_absolute_error(
    test["target"],
    test["prediction"]
))

print("\nOverall RMSE:", np.sqrt(
    mean_squared_error(
        test["target"],
        test["prediction"]
    )
))

print("\n=======================================")
