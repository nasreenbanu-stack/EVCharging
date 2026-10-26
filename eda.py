
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
file_path = Path("data/charging_records_clean.csv")
df = pd.read_csv(file_path)

# Convert start datetime into a datetime value
df["StartDatetime"] = pd.to_datetime(
    df["StartDatetime"], errors="coerce"
)

# Create useful time features for analysis
df["StartHour"] = df["StartDatetime"].dt.hour
df["StartMonth"] = df["StartDatetime"].dt.month

# Create an output folder for charts
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("=" * 55)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# 1. Overall demand statistics
print("\n1. DEMAND SUMMARY")
print(df["Demand"].describe().round(2))

# 2. Number of sessions and average demand by location
print("\n2. DEMAND BY LOCATION")
location_summary = (
    df.groupby("Location")
    .agg(
        Sessions=("Demand", "size"),
        AverageDemand=("Demand", "mean"),
        MedianDemand=("Demand", "median")
    )
    .sort_values("Sessions", ascending=False)
    .round(2)
)
print(location_summary)

# 3. Sessions and average demand by charger type
print("\n3. DEMAND BY CHARGER TYPE")
charger_summary = (
    df.groupby("ChargerType")
    .agg(
        Sessions=("Demand", "size"),
        AverageDemand=("Demand", "mean"),
        MedianDemand=("Demand", "median")
    )
    .round(2)
)
print(charger_summary)

# 4. Number of sessions by starting hour
print("\n4. SESSIONS BY START HOUR")
hour_summary = (
    df.groupby("StartHour")
    .size()
    .reindex(range(24), fill_value=0)
)
print(hour_summary)

# Chart 1: Demand distribution
plt.figure(figsize=(9, 5))
plt.hist(df["Demand"].dropna(), bins=30, edgecolor="black")
plt.title("Distribution of EV Charging Demand")
plt.xlabel("Demand")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig(output_dir / "demand_distribution.png", dpi=150)
plt.close()

# Chart 2: Number of sessions by location
location_summary["Sessions"].sort_values().plot(
    kind="barh", figsize=(10, 6)
)
plt.title("EV Charging Sessions by Location")
plt.xlabel("Number of Sessions")
plt.ylabel("Location")
plt.tight_layout()
plt.savefig(output_dir / "sessions_by_location.png", dpi=150)
plt.close()

# Chart 3: Average demand by location
location_summary["AverageDemand"].sort_values().plot(
    kind="barh", figsize=(10, 6)
)
plt.title("Average EV Charging Demand by Location")
plt.xlabel("Average Demand")
plt.ylabel("Location")
plt.tight_layout()
plt.savefig(output_dir / "average_demand_by_location.png", dpi=150)
plt.close()

# Chart 4: Sessions by starting hour
hour_summary.plot(kind="bar", figsize=(10, 5))
plt.title("EV Charging Sessions by Starting Hour")
plt.xlabel("Hour of Day (0 = midnight, 12 = noon)")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / "sessions_by_hour.png", dpi=150)
plt.close()

print("\nEDA completed successfully.")
print("Charts saved in the outputs folder:")
print("- demand_distribution.png")
print("- sessions_by_location.png")
print("- average_demand_by_location.png")
print("- sessions_by_hour.png")
