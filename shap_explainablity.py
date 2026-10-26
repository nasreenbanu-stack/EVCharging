
from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

print("=" * 60)
print("STEP 16: SHAP EXPLAINABILITY")
print("=" * 60)

# Load the saved model and training data
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

# Use a sample to keep the analysis manageable
sample = train_df[features].sample(
    n=min(300, len(train_df)),
    random_state=42,
)

# Apply the same preprocessing used by the model
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(sample)
feature_names = preprocessor.get_feature_names_out()

# Convert sparse matrices to dense arrays when necessary
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = pd.DataFrame(
    X_transformed,
    columns=feature_names,
)

print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_transformed)

# For binary classification, explain the High Demand class.
# SHAP output shape can vary by version.
if isinstance(shap_values, list):
    high_demand_values = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    high_demand_values = shap_values[:, :, 1]
else:
    high_demand_values = shap_values

Path("outputs").mkdir(exist_ok=True)

# Global explanation: which features matter most overall?
plt.figure()
shap.summary_plot(
    high_demand_values,
    X_transformed,
    feature_names=feature_names,
    show=False,
    max_display=15,
)
plt.tight_layout()
plt.savefig(
    "outputs/shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Saved outputs/shap_summary.png")

# Mean absolute SHAP values for a ranked table
mean_abs_shap = abs(high_demand_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "MeanAbsoluteSHAP": mean_abs_shap,
}).sort_values("MeanAbsoluteSHAP", ascending=False)

print("\nTOP 15 FEATURES BY MEAN ABSOLUTE SHAP VALUE")
print(shap_importance.head(15).round(4).to_string(index=False))

shap_importance.to_csv(
    "outputs/shap_importance.csv",
    index=False,
)

print("\nSaved outputs/shap_importance.csv")
print("\nStep 16 completed successfully.")
