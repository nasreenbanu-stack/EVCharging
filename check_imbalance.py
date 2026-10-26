
import pandas as pd

print("=" * 55)
print("STEP 13: CHECK CLASS IMBALANCE")
print("=" * 55)

# Load the classification training data only
df = pd.read_csv("data/processed/classification_train.csv")

# Match the chronological split used in Step 12
split_index = int(len(df) * 0.80)
fit_df = df.iloc[:split_index].copy()
validation_df = df.iloc[split_index:].copy()

for name, data in [
    ("Model-fitting training data", fit_df),
    ("Validation data", validation_df),
]:
    counts = data["DemandClass"].value_counts().sort_index()
    percentages = (
        data["DemandClass"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )

    print(f"\n{name}:")
    print(f"Normal Demand (0): {counts.get(0, 0):,} "
          f"({percentages.get(0, 0):.2f}%)")
    print(f"High Demand (1):   {counts.get(1, 0):,} "
          f"({percentages.get(1, 0):.2f}%)")

    if len(counts) == 2:
        ratio = counts.max() / counts.min()
        print(f"Majority/minority ratio: {ratio:.3f}:1")

print("\nStep 13 completed successfully.")
