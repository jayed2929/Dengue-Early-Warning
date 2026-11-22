
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# DENGUEGUARD AI - HISTORICAL DATA DASHBOARD
# HOUR 2: CITY, WARD AND WEATHER EXPLORATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hackathon_preli_dataset.xlsx"
)


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


# Load dataset
try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🦟 DengueGuard AI")
st.subheader("Urban Dengue Historical Monitoring Dashboard")

st.info(
    "This dashboard currently displays historical observations. "
    "It does not yet generate machine-learning predictions."
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
    "Dengue Early Warning| Historical data exploration | "
    "Machine-learning predictions will be integrated separately."
)



