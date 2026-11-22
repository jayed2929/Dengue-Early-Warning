# 🦟 DengueGuard AI

## AI-Powered Urban Dengue Early-Warning System

DengueGuard AI is a machine-learning prototype developed for the **EDU
AI Engineering Hackathon 2026** by **Team CoderGEN**. It uses historical
dengue cases, weather patterns, and urban vulnerability indicators to
estimate ward-level dengue risk.

The model estimates **cumulative dengue cases across weeks +2, +3, and
+4**. The interactive dashboard allows users to explore ward-level
estimates, compare risk categories, and inspect historical dengue and
weather patterns.

> **Important limitation:** This is an experimental research prototype
> based on historical data. The latest available observations are from
> **December 29, 2025**. Dashboard predictions are not live forecasts,
> official public-health alerts, or a substitute for public-health
> decisions.

------------------------------------------------------------------------

## 1. Problem and Project Objective

Dengue risk can vary across urban wards and over time. DengueGuard AI
explores whether historical dengue counts, multi-week weather patterns,
and urban vulnerability indicators can help identify differences in
potential ward-level risk.

The project aims to: - Engineer multi-week weather features from
historical observations. - Estimate cumulative dengue cases for the +2
to +4 week horizon. - Group model estimates into experimental Low,
Medium, High, and Critical risk categories. - Present results in an
interactive dashboard for exploration.

## 2. Team

**Team Name: CoderGEN**

| Member | Designation | Roles and Responsibilities |
|---|---|---|
| Mohammad Abu Jayed | Team Leader & Lead Developer | Led the project from development to submission, including dataset analysis, preprocessing, feature engineering, machine learning model development, model evaluation, Streamlit dashboard development, GitHub repository management, system integration, documentation, and final presentation preparation. |
| Prionta Karmakar | Team Member – Data Research & Initial Analysis | Participated in the initial project research and brainstorming during Hour 1, contributed to understanding the problem statement, identifying the project objectives, and discussing the initial solution approach. |
| Prome Dey | Team Member – Problem Analysis & Initial Planning | Participated in the initial problem analysis and brainstorming during Hour 1, contributed to discussing the project scope, identifying key challenges, and planning the initial development approach. |

## 3. Key Features

-   Ward-level dengue case estimation using a Random Forest model.
-   Multi-week rainfall lag and rolling-window feature engineering.
-   Temperature and relative humidity analysis.
-   Urban vulnerability indicators.
-   Four-level experimental risk classification: Low, Medium, High, and
    Critical.
-   Ranked ward-level prediction table.
-   Interactive city and ward filters.
-   Historical dengue and weather trend analysis.
-   Model evaluation using Mean Absolute Error (MAE) and Root Mean
    Squared Error (RMSE).

## 4. Technology Stack

  Component               Technology
  ----------------------- ----------------------------------------------
  Programming language    Python
  Data processing         Pandas, NumPy
  Machine learning        Scikit-learn
  Model serialization     Joblib
  Dashboard               Streamlit
  Charts                  Plotly
  Excel dataset reading   OpenPyXL
  Version control         Git and GitHub
  UI styling              Streamlit theme configuration and custom CSS

## 5. Project Structure

``` text
Dengue-Early-Warning/
├── .streamlit/
│   └── config.toml
├── assets/
│   └── style.css
├── data/
│   ├── raw/
│   │   └── hackathon_preli_dataset.xlsx
│   └── processed/
│       └── weekly_features.csv
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_baseline.py
│   ├── model_diagnostics.py
│   ├── risk_evaluation.py
│   ├── monthly_prediction_analysis.py
│   └── data_audit.py
├── models/
│   └── dengue_model_compressed.joblib
├── docs/
│   ├── data_audit_report.txt
│   ├── dataset_dictionary.md
│   └── weather_features.md
├── app.py
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

The `.venv/` virtual environment is created locally and is not committed
to the repository.

## 6. Installation and Setup

### Prerequisites

-   Git
-   Python (the project was tested with Python 3.14.3)
-   pip

### Step 1: Clone the repository

``` bash
git clone https://github.com/jayed2929/Dengue-Early-Warning.git
cd Dengue-Early-Warning
```

### Step 2: Create a virtual environment

``` bash
python -m venv .venv
```

### Step 3: Activate the environment

**Windows PowerShell:**

``` powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

``` cmd
.venv\Scripts\activate.bat
```

**Git Bash:**

``` bash
source .venv/Scripts/activate
```

### Step 4: Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Run the dashboard

``` bash
streamlit run app.py
```

Streamlit normally opens the dashboard at:

``` text
http://localhost:8501
```

To stop the local server, focus the terminal running Streamlit and press
`Ctrl + C`.

### Troubleshooting

-   Run commands from the repository root, where `app.py` and
    `requirements.txt` are located.
-   Confirm the virtual environment is activated before installing
    packages or launching the dashboard.
-   If dependencies are missing, run
    `python -m pip install -r requirements.txt` again.
-   The repository includes the processed dataset and compressed model
    required by the dashboard; model training is not required just to
    launch the dashboard.

## 7. Machine Learning Pipeline

The system uses a **Random Forest Regressor** to estimate cumulative
dengue cases across weeks +2, +3, and +4.

### Input features

-   Weekly rainfall lags: 1--4 weeks.
-   Rolling rainfall totals: 2, 3, and 4 weeks.
-   Four-week average temperature.
-   Four-week average relative humidity.
-   Population density.
-   Waterlogging risk index.
-   Sanitation score.
-   Seasonal month features.

### Prediction target

``` text
target_cases_2to4w =
    dengue_cases(+2)
    + dengue_cases(+3)
    + dengue_cases(+4)
```

This is one cumulative prediction over the +2 to +4 week window, not
three separate weekly predictions.

### Model configuration

  Parameter                                      Value
  -------------------------- -------------------------
  Algorithm                    Random Forest Regressor
  Number of estimators                             300
  Maximum depth                                     18
  Minimum samples per leaf                           2
  Random state                                      42

## 8. Model Evaluation

The model was evaluated using chronological training, validation, and
test periods.

  Metric     Validation    Test
  -------- ------------ -------
  MAE             2.432   3.359
  RMSE            4.451   6.102

These are retrospective results on the available dataset. They do not
establish real-world outbreak prediction accuracy or guarantee future
performance.

## 9. Risk Classification

  Risk level     Predicted cumulative cases
  ------------ ----------------------------
  Low                           Less than 1
  Medium                  1 to less than 14
  High                   14 to less than 28
  Critical                      28 or above

The thresholds are derived from training-data quantiles. They are
experimental and are **not officially validated public-health alert
thresholds**.

## 10. Dashboard Guide

The dashboard provides: - An overview of ward-level model estimates and
risk categories. - A ranked table of wards by estimated cumulative
cases. - City and ward filters. - A historical dengue trend chart. - A
selectable weather trend chart for rainfall, humidity, and
temperature. - Model evaluation information.

The dashboard uses historical observations. Selecting a ward or city
filters the displayed information; it does not make the underlying
dataset live.

## 11. Efficiency and Runtime

-   The trained model is serialized using Joblib.
-   The compressed model is loaded for inference; the dashboard does not
    retrain the model on each run.
-   Streamlit caching is used to reduce repeated data and model loading.
-   Prediction inference runs locally.
-   The current dashboard does not depend on external LLM or prediction
    APIs.

## 12. AI Tools and External API Disclosure

**AI assistant:** ChatGPT by OpenAI was used as an AI-assisted
development resource for explanations, debugging support, code review,
and documentation assistance.

**Model endpoint / external AI APIs:** The current application does not
call an external LLM endpoint or external prediction API. Dengue
predictions are generated locally using the project's trained Random
Forest model.

The team should add any other AI assistants, AI coding tools, model
endpoints, or external APIs actually used by any member, together with
their purpose.

## 13. Attributions and Data Sources

-   **Dataset:** `hackathon_preli_dataset.xlsx`, supplied for the EDU AI
    Engineering Hackathon. Add the organizer's official dataset
    attribution or license wording here if it was provided separately.
-   **Libraries and frameworks:** Python packages listed in
    `requirements.txt` are third-party open-source dependencies; their
    respective licenses apply.
-   **Code and visual assets:** Add attribution for any third-party code
    snippets, templates, icons, images, or other assets incorporated
    into the project. If none were used, state that only after verifying
    it.

## 14. Demo and Access

-   **Source repository:**
    https://github.com/jayed2929/Dengue-Early-Warning
-   **Run locally:** Follow the installation instructions in Section 6.
-   **Demo video:** \[Add the final walkthrough video link\]
-   **Deployed dashboard:** \[Add deployed URL if available; otherwise
    write "Not deployed --- run locally using the instructions above."\]

## 15. Limitations

-   The latest dataset observations are from December 29, 2025.
-   Current dashboard predictions are based on historical observations,
    not live 2026 weather.
-   Risk thresholds require validation by public-health experts.
-   Model performance may vary across wards, cities, and seasons.
-   The estimates should not replace official public-health
    surveillance, diagnosis, or decisions.

## 16. Project Status

Developed as an experimental prototype for the **EDU AI Engineering
Hackathon 2026** by **Team CoderGEN**.

**Repository:** https://github.com/jayed2929/Dengue-Early-Warning
