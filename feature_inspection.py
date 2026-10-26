
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("data/charging_records_clean.csv")

print("=" * 55)
print("STEP 5: FEATURE INSPECTION")
print("=" * 55)

# 1. Examine the relationship between Duration and Demand
print("\n1. DURATION AND DEMAND RELATIONSHIP")

print(
    "Correlation between Duration and Demand:",
    round(df["Duration"].corr(df["Demand"]), 4)
)

# Calculate demand per minute where duration is positive
positive_duration = df[df["Duration"] > 0].copy()
positive_duration["DemandPerMinute"] = (
    positive_duration["Demand"] / positive_duration["Duration"]
)

print("\nDemand / Duration ratio for positive durations:")
print(positive_duration["DemandPerMinute"].describe().round(5))

# 2. Show example records
print("\n2. SAMPLE DURATION AND DEMAND VALUES")
print(df[["Duration", "Demand"]].head(10).to_string(index=False))

# 3. Check the number of distinct values in each column
print("\n3. DISTINCT VALUES PER COLUMN")
print(df.nunique().sort_values())

# 4. Check how much data is available by starting hour
df["StartDatetime"] = pd.to_datetime(
    df["StartDatetime"], errors="coerce"
)
df["StartHour"] = df["StartDatetime"].dt.hour

print("\n4. AVERAGE DEMAND BY STARTING HOUR")
print(
    df.groupby("StartHour")["Demand"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# 5. Save a chart to inspect Duration versus Demand
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))
plt.scatter(
    df["Duration"],
    df["Demand"],
    alpha=0.25,
    s=10
)
plt.title("Duration versus EV Charging Demand")
plt.xlabel("Duration")
plt.ylabel("Demand")
plt.tight_layout()
plt.savefig(
    output_dir / "duration_vs_demand.png",
    dpi=150
)
plt.close()

print("\nFeature inspection completed.")
print("Chart saved to outputs/duration_vs_demand.png")
print("No dataset values were modified.")
