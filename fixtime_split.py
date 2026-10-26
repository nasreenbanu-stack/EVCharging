
from pathlib import Path
import pandas as pd

print("=" * 55)
print("STEP 9: STRICT CHRONOLOGICAL DATA SPLIT")
print("=" * 55)

df = pd.read_csv("data/charging_records_clean.csv")

df["StartDatetime"] = pd.to_datetime(
    df["StartDatetime"], errors="coerce"
)
df = df.dropna(subset=["StartDatetime"])
df = df.sort_values("StartDatetime").reset_index(drop=True)

# Find a timestamp near the 80% boundary
split_index = int(len(df) * 0.80)
cutoff_time = df.loc[split_index, "StartDatetime"]

# Keep every record with the same timestamp together
train_df = df[df["StartDatetime"] < cutoff_time].copy()
test_df = df[df["StartDatetime"] >= cutoff_time].copy()

# Create features available at the start of a session
for part in (train_df, test_df):
    part["StartHour"] = part["StartDatetime"].dt.hour
    part["StartDayOfWeek"] = part["StartDatetime"].dt.dayofweek
    part["StartMonth"] = part["StartDatetime"].dt.month
    part["StartDayOfMonth"] = part["StartDatetime"].dt.day
    part["IsWeekend"] = (
        part["StartDatetime"].dt.dayofweek >= 5
    ).astype(int)

features = [
    "Location",
    "ChargerType",
    "ChargerCompany",
    "StartHour",
    "StartDayOfWeek",
    "StartMonth",
    "StartDayOfMonth",
    "IsWeekend",
]

# Save only the features and target used by our models
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_df[features + ["Demand"]].to_csv(
    output_dir / "train.csv", index=False
)
test_df[features + ["Demand"]].to_csv(
    output_dir / "test.csv", index=False
)

print("\nCutoff timestamp:", cutoff_time)
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training ends:", train_df["StartDatetime"].max())
print("Testing starts:", test_df["StartDatetime"].min())

print(
    "Training and testing share a timestamp:",
    not set(train_df["StartDatetime"]).isdisjoint(
        set(test_df["StartDatetime"])
    )
)

print("\nUpdated train.csv and test.csv.")
print("Run step8_compare_regression.py again next.")
