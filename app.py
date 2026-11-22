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

# Load custom CSS
CSS_PATH = PROJECT_ROOT / "assets" / "style.css"

with open(CSS_PATH, encoding="utf-8") as css_file:
    st.markdown(
        f"<style>{css_file.read()}</style>",
        unsafe_allow_html=True
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

    
    else:
        # Rank wards by predicted dengue cases
        prediction_df = prediction_df.sort_values(
            "predicted_cases_2to4w",
            ascending=False
        ).copy()

        prediction_df["Rank"] = range(
            1, len(prediction_df) + 1
        )

        # Key prediction indicators
        total_wards = len(prediction_df)

        high_critical = prediction_df[
            prediction_df["risk_level"].isin(
                ["High", "Critical"]
            )
        ].shape[0]

        st.markdown("### Prediction Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Wards Assessed",
            total_wards
        )

        col2.metric(
            "High / Critical Risk",
            high_critical
        )

        col3.metric(
            "Prediction Period",
            "Weeks +2 to +4"
        )

        st.caption(
            "Estimates represent cumulative dengue cases "
            "across the prediction window, not weekly case counts."
        )

        # Ranked ward prediction table
        st.markdown("### Ward Risk Ranking")

        top_10 = prediction_df.head(10)[
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
                "predicted_cases_2to4w": "Estimated Cases",
                "risk_level": "Risk Level"
            }
        )

        top_10["Estimated Cases"] = (
            top_10["Estimated Cases"].round(2)
        )

        st.dataframe(
            top_10,
            width="stretch",
            hide_index=True
        )

        # Model performance
        with st.expander("Model Performance & Evaluation"):
            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Test MAE",
                f"{model_bundle['test_mae']:.3f}"
            )

            metric2.metric(
                "Test RMSE",
                f"{model_bundle['test_rmse']:.3f}"
            )

            st.caption(
                "These metrics are from retrospective test data. "
                "They do not guarantee future prediction accuracy."
            )

    
except FileNotFoundError as error:
    st.error(
        f"Required model or processed data file not found: {error}"
    )

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


# ============================================================
# DENGUE CASES OVER TIME
# ============================================================

st.markdown("## Historical Dengue Trend")

if selected_ward == "All Wards":
    # Aggregate weekly dengue cases across the selected city/cities
    dengue_trend = (
        filtered_df.groupby("date", as_index=False)
        ["dengue_cases"].sum()
    )

    chart_title = "Total Weekly Dengue Cases"
else:
    # Show only the selected ward's historical cases
    dengue_trend = (
        filtered_df.groupby("date", as_index=False)
        ["dengue_cases"].sum()
    )

    chart_title = f"Weekly Dengue Cases — {selected_ward}"

dengue_chart = px.line(
    dengue_trend,
    x="date",
    y="dengue_cases",
    title=chart_title,
    markers=False,
    labels={
        "date": "Week",
        "dengue_cases": "Dengue Cases"
    }
)

dengue_chart.update_traces(
    line=dict(color="#0D9488", width=2.5)
)

dengue_chart.update_layout(
    height=380,
    showlegend=False,
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(
        title="Number of Dengue Cases",
        rangemode="tozero",
        gridcolor="#E2E8F0"
    )
)

st.plotly_chart(
    dengue_chart,
    use_container_width=True
)




# ============================================================
# WEATHER TRENDS
# ============================================================

st.markdown("## Weather Trends")

# Aggregate weather values by date across the selected wards
weather_trend = (
    filtered_df.groupby("date", as_index=False)
    .agg({
        "rainfall_mm": "mean",
        "relative_humidity_pct": "mean",
        "avg_temp_c": "mean"
    })
)

weather_options = {
    "Rainfall": {
        "column": "rainfall_mm",
        "label": "Average Weekly Rainfall (mm)"
    },
    "Humidity": {
        "column": "relative_humidity_pct",
        "label": "Average Relative Humidity (%)"
    },
    "Temperature": {
        "column": "avg_temp_c",
        "label": "Average Temperature (°C)"
    }
}

selected_weather = st.selectbox(
    "Select Weather Indicator",
    list(weather_options.keys())
)

weather_config = weather_options[selected_weather]

weather_chart = px.line(
    weather_trend,
    x="date",
    y=weather_config["column"],
    title=f"{selected_weather} Trend Over Time",
    labels={
        "date": "Week",
        weather_config["column"]: weather_config["label"]
    }
)

weather_chart.update_traces(
    line=dict(color="#0D9488", width=2)
)

weather_chart.update_layout(
    height=350,
    showlegend=False,
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(
        title=weather_config["label"],
        gridcolor="#E2E8F0"
    )
)

st.plotly_chart(
    weather_chart,
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



