# 🦟 Dengue-Early-Warning
### AI-Powered Urban Dengue Early-Warning System

DengueGuard AI is a machine-learning-based prototype designed to estimate ward-level dengue risk using historical dengue cases, weather patterns, and urban vulnerability indicators.

The system analyzes multi-week weather patterns and predicts cumulative dengue cases across weeks +2, +3, and +4, helping users explore potential differences in dengue risk between urban wards.

> **Important:** This is an experimental research prototype using historical data. It does not provide live forecasts or officially validated public-health alerts.

---

## 1. Key Features

- Ward-level dengue risk prediction using machine learning.
- Multi-week rainfall lag and rolling-window feature engineering.
- Temperature and relative humidity analysis.
- Urban vulnerability indicators.
- Four-level risk classification: Low, Medium, High, and Critical.
- Top 10 highest-predicted ward ranking.
- Interactive ward-level prediction visualization.
- Historical dengue and weather trend analysis.
- City and ward filtering.
- Model evaluation using MAE and RMSE.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Dataset Reading | OpenPyXL |

## 3. Project Structure

```text
Dengue-Early-Warning/
│
├── data/
│   ├── raw/
│   │   └── hackathon_preli_dataset.xlsx
│   └── processed/
│       └── weekly_features.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_baseline.py
│   ├── model_diagnostics.py
│   ├── risk_evaluation.py
│   ├── monthly_prediction_analysis.py
│   └── data_audit.py
│
├── models/
│   └── dengue_model_compressed.joblib
│
├── docs/
│   ├── data_audit_report.txt
│   ├── dataset_dictionary.md
│   └── weather_features.md
│
├── app.py
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

## 4. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/jayed2929/Dengue-Early-Warning.git
cd Dengue-Early-Warning
```

### Step 2: Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Launch the dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser, normally at:

```text
http://localhost:8501
```

## 5. Machine Learning Pipeline

The system uses a Random Forest Regressor to estimate cumulative dengue cases across weeks +2, +3, and +4.

### Weather and urban features

- Weekly rainfall lags: 1–4 weeks.
- Rolling rainfall totals: 2, 3 and 4 weeks.
- Four-week average temperature.
- Four-week average relative humidity.
- Population density.
- Waterlogging risk index.
- Sanitation score.
- Seasonal month features.

### Prediction target

```text
target_cases_2to4w =
    dengue_cases(+2)
    + dengue_cases(+3)
    + dengue_cases(+4)
```

### Model configuration

- Algorithm: Random Forest Regressor
- Number of estimators: 300
- Maximum depth: 18
- Minimum samples per leaf: 2
- Random state: 42

## 6. Model Evaluation

The model was evaluated using chronological training, validation, and testing periods.

| Metric | Validation | Test |
|---|---:|---:|
| MAE | 2.432 | 3.359 |
| RMSE | 4.451 | 6.102 |

These metrics represent retrospective performance on the available dataset. They do not establish real-world outbreak prediction accuracy.

## 7. Risk Classification

| Risk Level | Predicted Cumulative Cases |
|---|---:|
| Low | Less than 1 |
| Medium | 1 to less than 14 |
| High | 14 to less than 28 |
| Critical | 28 or above |

The thresholds are derived from training-data quantiles. They are experimental and are not officially validated public-health alert thresholds.

## 8. Dashboard

The Streamlit dashboard provides:

- Historical dengue case monitoring.
- Weather trend visualization.
- City and ward filtering.
- Ward-level predicted dengue cases.
- Top 10 ward ranking.
- Ward-level prediction comparison.
- Risk-category summaries.
- Model performance metrics.

## 9. Efficiency

- The trained model is serialized using Joblib.
- The compressed model is loaded without retraining during dashboard execution.
- Streamlit caching reduces repeated data and model loading.
- Prediction inference runs locally.

## 10. AI Assistance and External APIs

**AI assistance:** ChatGPT (OpenAI) was used as an AI-assisted development and debugging resource.

**External APIs:** The current prediction and dashboard pipeline does not require external LLM or prediction APIs.

Update this section if additional AI assistants, APIs, or external services were used by any team member.

## 11. Limitations

- The dataset's latest observations are from December 2025.
- Current dashboard predictions are based on historical observations, not live 2026 weather.
- Risk thresholds require public-health validation.
- Model performance may vary across wards, cities, and seasons.
- Predictions should not replace official public-health decisions.

## 12. Project Status

Developed as a prototype for the EDU AI Engineering Hackathon 2026.

**Repository:** https://github.com/jayed2929/Dengue-Early-Warning
