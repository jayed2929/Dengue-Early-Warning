import numpy as np
import joblib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# DENGUEGUARD AI - HISTORICAL DATA DASHBOARD
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hackathon_preli_dataset.xlsx"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "weekly_features.csv"
)

MODEL_PATH = PROJECT_ROOT / "models" / "dengue_model_compressed.joblib"


st.set_page_config(
    page_title="DengueGuard AI",
    page_icon="🦟",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = [
        "dengue_cases",
        "rainfall_mm",
        "avg_temp_c",
        "relative_humidity_pct",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df.sort_values("date")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_latest_predictions():
    # Load engineered weather features
    features_df = pd.read_csv(PROCESSED_DATA_PATH)
    features_df["date"] = pd.to_datetime(features_df["date"])

    # Recreate seasonal features used during model training
    month = features_df["date"].dt.month
    features_df["month_sin"] = np.sin(2 * np.pi * month / 12)
    features_df["month_cos"] = np.cos(2 * np.pi * month / 12)

    bundle = load_model()
    features = bundle["features"]
    thresholds = bundle["risk_thresholds"]

    # Keep rows with all required model inputs
    valid_df = features_df.dropna(subset=features).copy()

    # Select the latest available observation for each ward
    latest = (
        valid_df.sort_values("date")
        .groupby("ward_id", as_index=False)
        .tail(1)
        .copy()
    )

    # Generate predictions


    predictions = bundle["model"].predict(latest[features])
    latest["predicted_cases_2to4w"] = np.maximum(predictions, 0)

    
    
    # Assign risk categories
    latest["risk_level"] = np.select(
        [
            latest["predicted_cases_2to4w"] < thresholds["medium"],
            latest["predicted_cases_2to4w"] < thresholds["high"],
            latest["predicted_cases_2to4w"] < thresholds["critical"],
        ],
        ["Low", "Medium", "High"],
        default="Critical",
    )

    # Debug predictions

    return latest, bundle




# Load dataset
try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🦟 Dengue Early Warning")
st.subheader("Urban Dengue Historical Monitoring Dashboard")

st.info(
    "Model estimates are based on the latest available historical "
    "observation (December 29, 2025). The model predicts cumulative "
    "dengue cases across weeks +2, +3 and +4. These are experimental "
    "historical-data-based estimates, not live forecasts."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Dashboard Filters")

cities = ["All Cities"] + sorted(
    df["city"].dropna().unique().tolist()
)

selected_city = st.sidebar.selectbox(
    "Select City",
    cities
)

filtered_df = df.copy()

if selected_city != "All Cities":
    filtered_df = filtered_df[
        filtered_df["city"] == selected_city
    ]


wards = ["All Wards"] + sorted(
    filtered_df["ward_id"].dropna().unique().tolist(),
    key=str
)

selected_ward = st.sidebar.selectbox(
    "Select Ward",
    wards
)

if selected_ward != "All Wards":
    filtered_df = filtered_df[
        filtered_df["ward_id"] == selected_ward
    ]


# ============================================================
# MACHINE LEARNING PREDICTIONS
# ============================================================

st.markdown("---")
st.markdown("## 🤖 AI Dengue Risk Predictions")

st.info(
    "The model estimates cumulative dengue cases across "
    "weeks +2, +3 and +4 using the latest available "
    "historical weather observations."
)

try:
    prediction_df, model_bundle = load_latest_predictions()

    # Apply the dashboard's city and ward filters
    if selected_city != "All Cities":
        prediction_df = prediction_df[
            prediction_df["city"] == selected_city
        ]

    if selected_ward != "All Wards":
        prediction_df = prediction_df[
            prediction_df["ward_id"] == selected_ward
        ]

    if prediction_df.empty:
        st.warning("No prediction data available for this selection.")
    else:
                # Rank wards by predicted dengue cases
        prediction_df = prediction_df.sort_values(
            "predicted_cases_2to4w",
            ascending=False
        ).copy()

        prediction_df["Rank"] = range(1, len(prediction_df) + 1)

        # Risk summary
        risk_counts = (
            prediction_df["risk_level"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Critical"],
                fill_value=0
            )
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Low Risk", int(risk_counts["Low"]))
        col2.metric("Medium Risk", int(risk_counts["Medium"]))
        col3.metric("High Risk", int(risk_counts["High"]))
        col4.metric("Critical Risk", int(risk_counts["Critical"]))

        st.markdown("### Ward-level Risk Predictions")
                # Top 10 highest predicted wards
        st.markdown("### 🏆 Top 10 Highest-Risk Wards")

        top_10 = prediction_df.head(10).copy()

        top_10_display = top_10[
            [
                "Rank",
                "city",
                "ward_id",
                "predicted_cases_2to4w",
                "risk_level"
            ]
        ].rename(
            columns={
                "city": "City",
                "ward_id": "Ward",
                "predicted_cases_2to4w": "Predicted Cases (+2 to +4 weeks)",
                "risk_level": "Risk Level"
            }
        )

        st.dataframe(
            top_10_display,
            width="stretch",
            hide_index=True
        )

                # Ward-level prediction comparison
        st.markdown("### 📊 Ward-level Risk Ranking")

        ranking_chart = px.bar(
            prediction_df.head(15),
            x="predicted_cases_2to4w",
            y="ward_id",
            color="city",
            orientation="h",
            hover_data=["risk_level", "date"],
            title="Predicted Dengue Cases by Ward",
            labels={
                "predicted_cases_2to4w": "Predicted Cumulative Cases (+2 to +4 weeks)",
                "ward_id": "Ward",
                "city": "City"
            }
        )

        ranking_chart.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=600
        )

        st.plotly_chart(
            ranking_chart,
            width="stretch"
        )

        display_predictions = prediction_df[
            [
                "date",
                "city",
                "ward_id",
                "predicted_cases_2to4w",
                "risk_level",
            ]
        ].sort_values(
            "predicted_cases_2to4w",
            ascending=False
        )

        display_predictions = display_predictions.rename(
            columns={
                "date": "Latest Observation",
                "city": "City",
                "ward_id": "Ward",
                "predicted_cases_2to4w":
                    "Predicted Cumulative Cases (+2 to +4 weeks)",
                "risk_level": "Risk Level",
            }
        )

        st.dataframe(
            display_predictions,
            use_container_width=True,
            hide_index=True
        )

        # Highest predicted risk
        highest_risk = prediction_df.iloc[
            prediction_df["predicted_cases_2to4w"].argmax()
        ]

        st.warning(
            f"Highest predicted risk: {highest_risk['city']} — "
            f"{highest_risk['ward_id']} | "
            f"Estimated cumulative cases: "
            f"{highest_risk['predicted_cases_2to4w']:.1f} | "
            f"Risk: {highest_risk['risk_level']}"
        )

        # Model performance
        st.markdown("### Model Performance")

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Test MAE",
            f"{model_bundle['test_mae']:.3f}"
        )

        metric2.metric(
            "Test RMSE",
            f"{model_bundle['test_rmse']:.3f}"
        )

except FileNotFoundError as error:
    st.error(f"Required model or processed data file not found: {error}")

except Exception as error:
    st.error(f"Prediction error: {error}")


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

st.markdown("## Historical Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Dengue Cases",
    f"{filtered_df['dengue_cases'].sum():,.0f}"
)

col2.metric(
    "Average Rainfall",
    f"{filtered_df['rainfall_mm'].mean():.2f} mm"
)

col3.metric(
    "Average Temperature",
    f"{filtered_df['avg_temp_c'].mean():.2f} °C"
)

col4.metric(
    "Average Humidity",
    f"{filtered_df['relative_humidity_pct'].mean():.2f}%"
)


# ============================================================
# DENGUE CASES OVER TIME
# ============================================================

st.markdown("## Dengue Case History")

dengue_chart = px.line(
    filtered_df,
    x="date",
    y="dengue_cases",
    color="ward_id" if selected_ward == "All Wards" else None,
    title="Historical Dengue Cases",
    labels={
        "date": "Date",
        "dengue_cases": "Dengue Cases",
        "ward_id": "Ward"
    }
)

st.plotly_chart(
    dengue_chart,
    use_container_width=True
)


# ============================================================
# WEATHER CHARTS
# ============================================================

st.markdown("## Weather Trends")

weather_col1, weather_col2 = st.columns(2)

with weather_col1:
    rainfall_chart = px.line(
        filtered_df,
        x="date",
        y="rainfall_mm",
        title="Weekly Rainfall",
        labels={
            "date": "Date",
            "rainfall_mm": "Rainfall (mm)"
        }
    )

    st.plotly_chart(
        rainfall_chart,
        use_container_width=True
    )


with weather_col2:
    humidity_chart = px.line(
        filtered_df,
        x="date",
        y="relative_humidity_pct",
        title="Relative Humidity",
        labels={
            "date": "Date",
            "relative_humidity_pct": "Humidity (%)"
        }
    )

    st.plotly_chart(
        humidity_chart,
        use_container_width=True
    )


# ============================================================
# TEMPERATURE CHART
# ============================================================

temperature_chart = px.line(
    filtered_df,
    x="date",
    y="avg_temp_c",
    title="Average Temperature Over Time",
    labels={
        "date": "Date",
        "avg_temp_c": "Average Temperature (°C)"
    }
)

st.plotly_chart(
    temperature_chart,
    use_container_width=True
)


# ============================================================
# DATA TABLE
# ============================================================

st.markdown("## Historical Records")

st.dataframe(
    filtered_df[
        [
            "date",
            "city",
            "ward_id",
            "dengue_cases",
            "rainfall_mm",
            "avg_temp_c",
            "relative_humidity_pct",
        ]
    ].sort_values("date", ascending=False),
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Risk thresholds are derived from training-data quantiles "
    "and are not officially validated public-health alert levels."
)



