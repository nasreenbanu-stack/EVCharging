
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("STEP 15: FEATURE IMPORTANCE")
print("=" * 60)

# Load the tuned model and training data
model = joblib.load(
    "models/gradient_boosting_tuned.joblib"
)

train_df = pd.read_csv(
    "data/processed/classification_train.csv"
)

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

X_train = train_df[features]

# Get transformed feature names after preprocessing
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

transformed_names = preprocessor.get_feature_names_out()
importance_values = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": transformed_names,
    "Importance": importance_values,
})

# Remove preprocessing prefixes for readability
importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("categorical__", "", regex=False)
    .str.replace("numeric__", "", regex=False)
)

importance_df = importance_df.sort_values(
    "Importance", ascending=False
)

print("\nTOP 15 MOST IMPORTANT FEATURES")
print(
    importance_df.head(15).round(4).to_string(index=False)
)

# Save complete feature importance table
Path("outputs").mkdir(exist_ok=True)

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False,
)

# Plot the top 15 features
top_features = importance_df.head(15).sort_values(
    "Importance", ascending=True
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"],
)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features — Tuned Gradient Boosting")
plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300,
)
plt.close()

print("\nSaved outputs/feature_importance.csv")
print("Saved outputs/feature_importance.png")
print("\nStep 15 completed successfully.")
