import pandas as pd
from pathlib import Path

# 1. Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. Locate the Excel dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hackathon_preli_dataset.xlsx"
)

# 3. Load the dataset
print("\n========== DENGUE DATASET AUDIT ==========\n")

df = pd.read_excel(DATA_PATH)

# 4. Dataset dimensions
print("1. DATASET SIZE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 5. Column names
print("\n2. COLUMN NAMES")
for column in df.columns:
    print("-", column)

# 6. Missing values
print("\n3. MISSING VALUES")
print(df.isnull().sum())

# 7. Duplicate records
print("\n4. DUPLICATE RECORDS")
print("Full duplicate rows:", df.duplicated().sum())

print(
    "Duplicate ward-date pairs:",
    df.duplicated(subset=["ward_id", "date"]).sum()
)

# 8. Unique cities and wards
print("\n5. GEOGRAPHICAL INFORMATION")

print("Number of cities:", df["city"].nunique())
print("Number of wards:", df["ward_id"].nunique())

print("\nRecords by city:")
print(
    df.groupby("city")
    .agg(
        wards=("ward_id", "nunique"),
        records=("ward_id", "size")
    )
)

# 9. Date range
print("\n6. DATE RANGE")

df["date"] = pd.to_datetime(df["date"])

print("First date:", df["date"].min())
print("Last date:", df["date"].max())

# 10. Dengue case statistics
print("\n7. DENGUE CASE STATISTICS")

print("Total dengue cases:", df["dengue_cases"].sum())
print("Average dengue cases:", df["dengue_cases"].mean())
print("Maximum dengue cases:", df["dengue_cases"].max())

print(
    "Zero-case records:",
    (df["dengue_cases"] == 0).sum()
)

# 11. Records per ward
print("\n8. RECORDS PER WARD")

ward_counts = df.groupby("ward_id").size()

print("Minimum records:", ward_counts.min())
print("Maximum records:", ward_counts.max())

# 12. Save report
REPORT_PATH = PROJECT_ROOT / "docs" / "data_audit_report.txt"

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("DENGUE DATASET AUDIT\n")
    file.write("====================\n\n")

    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n")
    file.write(f"Cities: {df['city'].nunique()}\n")
    file.write(f"Wards: {df['ward_id'].nunique()}\n")
    file.write(f"Missing values: {df.isnull().sum().sum()}\n")

    file.write(
        f"Duplicate ward-date pairs: "
        f"{df.duplicated(subset=['ward_id', 'date']).sum()}\n"
    )

    file.write(f"Total dengue cases: {df['dengue_cases'].sum()}\n")
    file.write(f"First date: {df['date'].min()}\n")
    file.write(f"Last date: {df['date'].max()}\n")

print("\nAudit completed successfully!")
print("Report saved to:", REPORT_PATH)