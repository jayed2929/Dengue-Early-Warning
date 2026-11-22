
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "weekly_features.csv"


df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["ward_id", "date"]).copy()

# Recreate the same future target.
grouped = df.groupby("ward_id")["dengue_cases"]

df["target"] = (
    grouped.shift(-2)
    + grouped.shift(-3)
    + grouped.shift(-4)
)

# Match the same historical feature eligibility.
feature_columns = [
    "rainfall_lag_1w",
    "rainfall_lag_2w",
    "rainfall_lag_3w",
    "rainfall_lag_4w",
    "rainfall_sum_2w",
    "rainfall_sum_3w",
    "rainfall_sum_4w",
    "avg_temp_c_avg_4w",
    "relative_humidity_pct_avg_4w",
]

df = df.dropna(
    subset=feature_columns + ["target"]
)

train = df[df["date"] <= "2023-11-27"].copy()

test = df[
    (df["date"] >= "2025-01-06")
    & (df["date"] <= "2025-11-24")
].copy()


def report(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    print(f"\n{name}")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))


# Baseline 1: Predict the training median for every ward.
global_median = train["target"].median()

global_predictions = np.full(
    len(test),
    global_median
)

report(
    "GLOBAL MEDIAN BASELINE",
    test["target"],
    global_predictions
)


# Baseline 2: Predict each ward's training median.
ward_medians = train.groupby("ward_id")["target"].median()

ward_predictions = test["ward_id"].map(ward_medians)

# Fallback to global median for any unseen ward.
ward_predictions = ward_predictions.fillna(global_median)

report(
    "WARD-SPECIFIC MEDIAN BASELINE",
    test["target"],
    ward_predictions
)


print("\nTest rows:", len(test))
print("Training median:", global_median)
print("Test target mean:", round(test["target"].mean(), 3))
print("Test target median:", test["target"].median())
