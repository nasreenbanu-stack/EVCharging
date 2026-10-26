
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

print("=" * 60)
print("STEP 19: FINAL REGRESSION EVALUATION")
print("=" * 60)

# Load the regression training and reserved test data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

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

target = "Demand"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

categorical_features = [
    "Location",
    "ChargerType",
    "ChargerCompany",
]

numeric_features = [
    "StartHour",
    "StartDayOfWeek",
    "StartMonth",
    "StartDayOfMonth",
    "IsWeekend",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numeric", "passthrough", numeric_features),
    ]
)

# Use the Gradient Boosting settings from Step 8
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
        ),
    ]
)

print(f"\nTraining records: {len(train_df):,}")
print(f"Test records: {len(test_df):,}")
print("\nTraining Gradient Boosting Regressor...")

# Fit on the full training dataset
model.fit(X_train, y_train)

# Evaluate once on the reserved test dataset
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nFINAL REGRESSION TEST RESULTS")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Save metrics
Path("outputs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

metrics_df = pd.DataFrame([{
    "Model": "Gradient Boosting",
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
}])

metrics_df.to_csv(
    "outputs/final_regression_metrics.csv",
    index=False,
)

# Save the trained model
joblib.dump(
    model,
    "models/gradient_boosting_regression_final.joblib",
)

# Plot actual versus predicted Demand
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.3)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Demand — Test Set")

plt.tight_layout()
plt.savefig(
    "outputs/regression_actual_vs_predicted.png",
    dpi=300,
)
plt.close()

print("\nSaved outputs/final_regression_metrics.csv")
print("Saved outputs/regression_actual_vs_predicted.png")
print("Saved models/gradient_boosting_regression_final.joblib")
print("\nStep 19 completed successfully.")
