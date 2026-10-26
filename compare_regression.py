
from pathlib import Path
import math
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

print("=" * 60)
print("STEP 8: REGRESSION MODEL COMPARISON")
print("=" * 60)

# Load training data only.
# Keep test.csv untouched for final evaluation.
df = pd.read_csv("data/processed/train.csv")

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

features = categorical_features + numerical_features
target = "Demand"

# The training file is already in chronological order.
# Use its earlier 80% for fitting and later 20% for validation.
split_index = int(len(df) * 0.80)

train_part = df.iloc[:split_index]
validation_part = df.iloc[split_index:]

X_train = train_part[features]
y_train = train_part[target]

X_val = validation_part[features]
y_val = validation_part[target]

print("\nTRAINING RECORDS:", len(X_train))
print("VALIDATION RECORDS:", len(X_val))
print("Final test records remain reserved:", 14566)

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numerical", "passthrough", numerical_features),
        ]
    )

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=20,
        random_state=42,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

results = []

for name, estimator in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("regressor", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    })

results_df = pd.DataFrame(results).sort_values("MAE")

print("\n" + "=" * 60)
print("VALIDATION RESULTS — LOWER MAE AND RMSE ARE BETTER")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

results_df.to_csv(
    output_dir / "regression_model_comparison.csv",
    index=False,
)

print("\nResults saved to:")
print("outputs/regression_model_comparison.csv")
print("\nStep 8 completed successfully.")
