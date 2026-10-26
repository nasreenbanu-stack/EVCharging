
from pathlib import Path
import pandas as pd

print("=" * 55)
print("STEP 6: PREPARE TRAINING AND TESTING DATA")
print("=" * 55)

# Load the cleaned dataset
input_file = Path("data/charging_records_clean.csv")
df = pd.read_csv(input_file)

# Convert the starting datetime
df["StartDatetime"] = pd.to_datetime(
    df["StartDatetime"], errors="coerce"
)

# Remove records without a valid start datetime, if any
before = len(df)
df = df.dropna(subset=["StartDatetime"]).copy()

print("\nRecords removed due to invalid start datetime:", before - len(df))

# Sort chronologically: earlier sessions first
df = df.sort_values("StartDatetime").reset_index(drop=True)

# Create features available at the start of a charging session
df["StartHour"] = df["StartDatetime"].dt.hour
df["StartDayOfWeek"] = df["StartDatetime"].dt.dayofweek
df["StartMonth"] = df["StartDatetime"].dt.month
df["StartDayOfMonth"] = df["StartDatetime"].dt.day
df["IsWeekend"] = (
    df["StartDatetime"].dt.dayofweek >= 5
).astype(int)

# Initial features for the prediction model
feature_columns = [
    "Location",
    "ChargerType",
    "ChargerCompany",
    "StartHour",
    "StartDayOfWeek",
    "StartMonth",
    "StartDayOfMonth",
    "IsWeekend",
]

target_column = "Demand"

# Keep only the selected features and target
model_data = df[feature_columns + [target_column]].copy()

# Chronological 80/20 split
split_index = int(len(model_data) * 0.80)

train_data = model_data.iloc[:split_index].copy()
test_data = model_data.iloc[split_index:].copy()

# Create the output folder
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save datasets
train_file = output_dir / "train.csv"
test_file = output_dir / "test.csv"

train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

# Report results
print("\nFEATURES SELECTED")
for feature in feature_columns:
    print("-", feature)

print("\nFEATURES EXCLUDED FROM INITIAL MODEL")
print("- UserID and ChargerID: identifiers")
print("- Duration: may not be known when predicting")
print("- EndDatetime, EndDay and EndTime: end-of-session information")
print("- Demand: target to predict, not an input feature")

print("\nDATASET SPLIT")
print("Total records:", len(model_data))
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTRAINING PERIOD")
print("From:", df.iloc[0]["StartDatetime"])
print("To:", df.iloc[split_index - 1]["StartDatetime"])

print("\nTESTING PERIOD")
print("From:", df.iloc[split_index]["StartDatetime"])
print("To:", df.iloc[-1]["StartDatetime"])

print("\nTARGET SUMMARY")
print("Training demand median:", round(train_data["Demand"].median(), 2))
print("Testing demand median:", round(test_data["Demand"].median(), 2))

print("\nSAVED FILES")
print(train_file)
print(test_file)

print("\nStep 6 completed successfully.")
