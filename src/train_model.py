
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# HOUR 3: DENGUE PREDICTION MODEL
# Target: Cumulative dengue cases in weeks 2, 3 and 4 ahead
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weekly_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "dengue_model.joblib"


# Chronological split with a four-week gap between periods.
TRAIN_END = pd.Timestamp("2023-11-27")

VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2024-11-25")

TEST_START = pd.Timestamp("2025-01-06")
TEST_END = pd.Timestamp("2025-11-24")


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["ward_id", "date"]
    ).reset_index(drop=True)

    print("Dataset loaded:", df.shape)

    return df


# ============================================================
# 2. CREATE FUTURE DENGUE TARGET
# ============================================================

def create_target(df):

    result = df.copy()

    grouped_cases = result.groupby(
        "ward_id"
    )["dengue_cases"]

    # Cases 2, 3 and 4 weeks after the current date.
    result["cases_plus_2w"] = grouped_cases.shift(-2)
    result["cases_plus_3w"] = grouped_cases.shift(-3)
    result["cases_plus_4w"] = grouped_cases.shift(-4)

    result["target_cases_2to4w"] = (
        result["cases_plus_2w"]
        + result["cases_plus_3w"]
        + result["cases_plus_4w"]
    )

    # Remove temporary future case columns.
    result = result.drop(
        columns=[
            "cases_plus_2w",
            "cases_plus_3w",
            "cases_plus_4w",
        ]
    )

    return result


# ============================================================
# 3. DEFINE INPUT FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "city",
    "ward_id",
]

NUMERIC_FEATURES = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",

    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",

    "rainfall_lag_1w",
    "rainfall_lag_2w",
    "rainfall_lag_3w",
    "rainfall_lag_4w",

    "rainfall_sum_2w",
    "rainfall_sum_3w",
    "rainfall_sum_4w",

    "avg_temp_c_avg_4w",
    "relative_humidity_pct_avg_4w",

    "month_sin",
    "month_cos",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET = "target_cases_2to4w"


def prepare_features(df):

    result = df.copy()

    # Month is known at prediction time.
    month = result["date"].dt.month

    result["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    result["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    # Check that all expected columns exist.
    required = FEATURES + [TARGET]

    missing = [
        col for col in required
        if col not in result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Do not use dengue_cases as an input feature.
    # It is used only to construct the future target.

    # Remove rows without enough past weather history
    # or without a complete future target.
    result = result.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    return result


# ============================================================
# 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(df):

    train = df[
        df["date"] <= TRAIN_END
    ].copy()

    validation = df[
        (df["date"] >= VALIDATION_START)
        & (df["date"] <= VALIDATION_END)
    ].copy()

    test = df[
        (df["date"] >= TEST_START)
        & (df["date"] <= TEST_END)
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "One of the chronological data splits is empty."
        )

    print("\nChronological data split")
    print("------------------------")
    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Testing rows:", len(test))

    print(
        "Training dates:",
        train["date"].min().date(),
        "to",
        train["date"].max().date()
    )

    print(
        "Validation dates:",
        validation["date"].min().date(),
        "to",
        validation["date"].max().date()
    )

    print(
        "Testing dates:",
        test["date"].min().date(),
        "to",
        test["date"].max().date()
    )

    return train, validation, test


# ============================================================
# 5. BUILD RANDOM FOREST PIPELINE
# ============================================================

def build_model():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", random_forest)
        ]
    )

    return model


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate_model(model, data, dataset_name):

    X = data[FEATURES]
    y_true = data[TARGET]

    predictions = model.predict(X)

    predictions = np.maximum(
        predictions,
        0
    )

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    print(f"\n{dataset_name} evaluation")
    print("------------------------")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    return predictions, mae, rmse


# ============================================================
# 7. RISK THRESHOLDS
# ============================================================

def assign_risk_tier(prediction, thresholds):

    if prediction < thresholds["medium"]:
        return "Low"

    elif prediction < thresholds["high"]:
        return "Medium"

    elif prediction < thresholds["critical"]:
        return "High"

    else:
        return "Critical"


# ============================================================
# 8. MAIN TRAINING WORKFLOW
# ============================================================

def main():

    print("\n========== DENGUE ML TRAINING ==========\n")

    df = load_data()

    # Create future dengue target.
    df = create_target(df)

    print("Future dengue target created.")

    # Prepare model features.
    df = prepare_features(df)

    print("Rows after removing incomplete features:", len(df))

    # Chronological split.
    train, validation, test = split_data(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    # Build and train model.
    model = build_model()

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # Evaluate validation set.
    validation_predictions, validation_mae, validation_rmse = (
        evaluate_model(
            model,
            validation,
            "VALIDATION"
        )
    )

    # Evaluate untouched test set.
    test_predictions, test_mae, test_rmse = (
        evaluate_model(
            model,
            test,
            "TEST"
        )
    )

    # Risk thresholds are calculated from training targets only.
    thresholds = {
        "medium": float(y_train.quantile(0.50)),
        "high": float(y_train.quantile(0.75)),
        "critical": float(y_train.quantile(0.90)),
    }

    print("\nRisk thresholds (training data)")
    print("-------------------------------")
    print(thresholds)

    # Show test predictions.
    results = test[
        ["date", "city", "ward_id", TARGET]
    ].copy()

    results["predicted_cases_2to4w"] = test_predictions

    results["risk_tier"] = results[
        "predicted_cases_2to4w"
    ].apply(
        lambda value: assign_risk_tier(
            value,
            thresholds
        )
    )

    print("\nSample test predictions")
    print("-----------------------")
    print(
        results.head(15).to_string(index=False)
    )

    # Save model and its metadata together.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "risk_thresholds": thresholds,
        "target_definition": (
            "Cumulative dengue cases at weeks +2, +3 and +4"
        ),
        "validation_mae": float(validation_mae),
        "validation_rmse": float(validation_rmse),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    print("\nModel saved successfully:")
    print(MODEL_PATH)

    print("\n========== TRAINING COMPLETED ==========\n")


if __name__ == "__main__":
    main()

