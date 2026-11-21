import streamlit as st
import pandas as pd
from pathlib import Path

# 1. Project directory
PROJECT_ROOT = Path(__file__).resolve().parent

# 2. Dataset location
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hackathon_preli_dataset.xlsx"
)

# 3. Page configuration
st.set_page_config(
    page_title="DengueGuard AI",
    page_icon="🦟",
    layout="wide"
)

# 4. Load dataset
@st.cache_data
def load_data():
    data = pd.read_excel(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])
    return data


# 5. Main application
st.title("🦟 DengueGuard AI")

st.subheader(
    "Urban Dengue Outbreak Monitoring & Early Warning System"
)

st.info(
    "Initial dashboard: displaying historical data. "
    "ML predictions will be integrated in a later stage."
)

# 6. Load data safely
try:
    df = load_data()

except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

# 7. Sidebar
st.sidebar.title("Dashboard Filters")

cities = sorted(df["city"].unique())

selected_city = st.sidebar.selectbox(
    "Select City",
    cities
)

city_data = df[df["city"] == selected_city]

wards = sorted(city_data["ward_id"].unique())

selected_ward = st.sidebar.selectbox(
    "Select Ward",
    wards
)

ward_data = city_data[
    city_data["ward_id"] == selected_ward
].sort_values("date")

# 8. Dashboard metrics
st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Total Cities", df["city"].nunique())

with col3:
    st.metric("Total Wards", df["ward_id"].nunique())

with col4:
    st.metric(
        "Total Dengue Cases",
        f"{df['dengue_cases'].sum():,}"
    )

# 9. Selected ward information
st.divider()

st.subheader("Selected Ward Analysis")

st.write("City:", selected_city)
st.write("Ward:", selected_ward)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Historical Cases",
        int(ward_data["dengue_cases"].sum())
    )

with col2:
    st.metric(
        "Average Rainfall",
        f"{ward_data['rainfall_mm'].mean():.2f} mm"
    )

with col3:
    st.metric(
        "Average Humidity",
        f"{ward_data['relative_humidity_pct'].mean():.2f}%"
    )

# 10. Dengue cases chart
st.subheader("Historical Dengue Cases")

st.line_chart(
    ward_data.set_index("date")["dengue_cases"]
)

# 11. Rainfall chart
st.subheader("Historical Rainfall")

st.bar_chart(
    ward_data.set_index("date")["rainfall_mm"]
)

# 12. Historical weather table
st.subheader("Weather Information")

st.dataframe(
    ward_data[
        [
            "date",
            "avg_temp_c",
            "max_temp_c",
            "min_temp_c",
            "rainfall_mm",
            "relative_humidity_pct",
            "dengue_cases"
        ]
    ].tail(20),
    use_container_width=True
)

# 13. Footer
st.divider()

st.caption(
    "DengueGuard AI | EDU AI Engineering Hackathon 2026"
)