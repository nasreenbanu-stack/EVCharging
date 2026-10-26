
from pathlib import Path
import pandas as pd

# File paths
input_file = Path("data/charging_records.csv")
output_file = Path("data/charging_records_clean.csv")

# Load original dataset
df = pd.read_csv(input_file)

print("=" * 50)
print("STEP 3: DATA CLEANING")
print("=" * 50)

original_rows = len(df)
print("\nOriginal number of rows:", original_rows)

# Step 1: Remove exact duplicate rows
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicates_removed)

# Step 2: Convert datetime columns
df["StartDatetime"] = pd.to_datetime(
    df["StartDatetime"], errors="coerce"
)
df["EndDatetime"] = pd.to_datetime(
    df["EndDatetime"], errors="coerce"
)

# Step 3: Identify records with invalid dates or durations
invalid_datetime = (
    df["StartDatetime"].isna()
    | df["EndDatetime"].isna()
)

negative_duration = df["Duration"] < 0
end_before_start = df["EndDatetime"] < df["StartDatetime"]

invalid_records = (
    invalid_datetime
    | negative_duration
    | end_before_start
)

invalid_removed = int(invalid_records.sum())
df = df.loc[~invalid_records].copy()

print("Invalid records removed:", invalid_removed)

# Step 4: Keep zero-duration records for now.
# We have not yet established why they occur.
print(
    "Zero-duration records retained:",
    int((df["Duration"] == 0).sum())
)

# Step 5: Save cleaned dataset separately
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print("\nCLEANING SUMMARY")
print("-" * 30)
print("Original rows:", original_rows)
print("Duplicates removed:", duplicates_removed)
print("Invalid records removed:", invalid_removed)
print("Final cleaned rows:", len(df))
print("Missing values after cleaning:", int(df.isnull().sum().sum()))
print("\nCleaned file saved to:", output_file)
print("Original dataset was not modified.")
