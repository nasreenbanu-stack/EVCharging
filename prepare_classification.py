
from pathlib import Path
import pandas as pd

print("=" * 55)
print("STEP 10: PREPARE CLASSIFICATION DATA")
print("=" * 55)

# Load the chronological datasets
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Calculate the threshold using training data only
threshold = train_df["Demand"].median()

# Create the classification target
train_df["DemandClass"] = (
    train_df["Demand"] >= threshold
).astype(int)

test_df["DemandClass"] = (
    test_df["Demand"] >= threshold
).astype(int)

# 1 = High Demand, 0 = Normal Demand
print(f"\nTraining median threshold: {threshold:.2f}")

print("\nTraining class counts:")
print(
    train_df["DemandClass"]
    .map({0: "Normal Demand", 1: "High Demand"})
    .value_counts()
)

print("\nTesting class counts:")
print(
    test_df["DemandClass"]
    .map({0: "Normal Demand", 1: "High Demand"})
    .value_counts()
)

# Save separate classification datasets
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(
    output_dir / "classification_train.csv",
    index=False
)

test_df.to_csv(
    output_dir / "classification_test.csv",
    index=False
)

print("\nSaved:")
print("data/processed/classification_train.csv")
print("data/processed/classification_test.csv")
print("\nStep 10 completed successfully.")
