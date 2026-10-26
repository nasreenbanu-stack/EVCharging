
from pathlib import Path
import pandas as pd

# Load the original dataset
file_path = Path("data/charging_records.csv")
df = pd.read_csv(file_path)

print("=" * 55)
print("STEP 2: DATA QUALITY INSPECTION")
print("=" * 55)

# 1. Check duplicate rows
print("\n1. EXACT DUPLICATE ROWS")
print("Number of duplicate rows:", df.duplicated().sum())

# 2. Check Duration
print("\n2. DURATION SUMMARY")
print(df["Duration"].describe())

print("\nNegative durations:", (df["Duration"] < 0).sum())
print("Zero durations:", (df["Duration"] == 0).sum())

# 3. Check Demand
print("\n3. DEMAND SUMMARY")
print(df["Demand"].describe())

print("\nNegative demand values:", (df["Demand"] < 0).sum())
print("Zero demand values:", (df["Demand"] == 0).sum())

# 4. Check date/time conversion without changing df
print("\n4. DATETIME VALIDATION")
start_dt = pd.to_datetime(df["StartDatetime"], errors="coerce")
end_dt = pd.to_datetime(df["EndDatetime"], errors="coerce")

print("Invalid start datetime values:", start_dt.isna().sum())
print("Invalid end datetime values:", end_dt.isna().sum())

print(
    "End datetime earlier than start datetime:",
    (end_dt < start_dt).sum()
)

# 5. Review categories
print("\n5. LOCATION VALUES")
print(df["Location"].value_counts())

print("\n6. CHARGER TYPE VALUES")
print(df["ChargerType"].value_counts().sort_index())

# 6. Review a few duplicate records
print("\n7. SAMPLE DUPLICATE RECORDS")
duplicates = df[df.duplicated(keep=False)]
print(duplicates.head(10).to_string(index=False))

print("\nInspection complete. Original dataset was not modified.")
