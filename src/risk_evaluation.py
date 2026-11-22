
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "weekly_features.csv"
MODEL_PATH = ROOT / "models" / "dengue_model.joblib"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["ward_id", "date"]).copy()

# Recreate the future target
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

# Load trained model
bundle = joblib.load(MODEL_PATH)

features = bundle["features"]
thresholds = bundle["risk_thresholds"]

df = df.dropna(subset=features + ["target"])

# Select chronological test period
test = df[
    (df["date"] >= "2025-01-06")
    & (df["date"] <= "2025-11-24")
].copy()

# Predict
predictions = bundle["model"].predict(test[features])
test["prediction"] = np.maximum(predictions, 0)

# Actual risk categories
def classify_risk(values):
    return np.select(
        [
            values < thresholds["medium"],
            values < thresholds["high"],
            values < thresholds["critical"],
        ],
        ["Low", "Medium", "High"],
        default="Critical",
    )

test["actual_risk"] = classify_risk(test["target"])
test["predicted_risk"] = classify_risk(test["prediction"])

labels = ["Low", "Medium", "High", "Critical"]

print("\n========== RISK CLASSIFICATION ==========")

print("\nAccuracy:")
print(accuracy_score(test["actual_risk"], test["predicted_risk"]))

print("\nClassification Report:")
print(
    classification_report(
        test["actual_risk"],
        test["predicted_risk"],
        labels=labels,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        test["actual_risk"],
        test["predicted_risk"],
        labels=labels
    )
)

print("\nActual risk distribution:")
print(test["actual_risk"].value_counts())

print("\nPredicted risk distribution:")
print(test["predicted_risk"].value_counts())

print("\n=========================================")
