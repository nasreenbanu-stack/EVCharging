
from pathlib import Path
import math
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib

print("=" * 55)
print("STEP 7: BASELINE REGRESSION MODEL")
print("=" * 55)

# Load the prepared datasets
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

target = "Demand"

categorical_features = [
    "Location",
    "ChargerType",
    "ChargerCompany",
]

numerical_features = [
    "StartHour",
    "StartDayOfWeek",
    "StartMonth",
    "StartDayOfMonth",
    "IsWeekend",
]

feature_columns = categorical_features + numerical_features

X_train = train_df[feature_columns]
y_train = train_df[target]

X_test = test_df[feature_columns]
y_test = test_df[target]

print("\n1. DATA LOADED")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Input features:", len(feature_columns))

# Encode categorical features.
# The encoder is fitted only on training data.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numerical", "passthrough", numerical_features),
    ]
)

# Build the linear regression pipeline
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# Fit the model using training data only
linear_model.fit(X_train, y_train)

# Predict demand for the unseen testing data
linear_predictions = linear_model.predict(X_test)

# Simple baseline: always predict training median
median_model = DummyRegressor(strategy="median")
median_model.fit(X_train, y_train)
median_predictions = median_model.predict(X_test)

def evaluate_model(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    r2 = r2_score(actual, predicted)

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

results = pd.DataFrame(
    [
        evaluate_model(
            "Median Baseline", y_test, median_predictions
        ),
        evaluate_model(
            "Linear Regression", y_test, linear_predictions
        ),
    ]
)

print("\n2. MODEL EVALUATION ON TEST DATA")
print(results.round(4).to_string(index=False))

# Save the trained model
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "linear_regression_baseline.joblib"
joblib.dump(linear_model, model_path)

# Save evaluation results
results_path = Path("outputs")
results_path.mkdir(parents=True, exist_ok=True)

results.to_csv(
    results_path / "baseline_regression_results.csv",
    index=False,
)

# Show sample predictions
comparison = pd.DataFrame(
    {
        "Actual_Demand": y_test.iloc[:10].values,
        "Predicted_Demand": linear_predictions[:10],
    }
)

print("\n3. FIRST 10 TEST PREDICTIONS")
print(comparison.round(2).to_string(index=False))

print("\nSaved model:", model_path)
print("Saved metrics: outputs/baseline_regression_results.csv")
print("\nStep 7 completed successfully.")
