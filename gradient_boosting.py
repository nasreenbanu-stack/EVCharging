
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

print("=" * 60)
print("STEP 14: TUNE GRADIENT BOOSTING")
print("=" * 60)

# Use training data only. The final test set stays untouched.
df = pd.read_csv("data/processed/classification_train.csv")

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

target = "DemandClass"

# Preserve chronological order
split_index = int(len(df) * 0.80)
train_df = df.iloc[:split_index]
valid_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df[target]
X_valid = valid_df[features]
y_valid = valid_df[target]

def make_pipeline(parameters):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )

    classifier = GradientBoostingClassifier(
        random_state=42,
        **parameters,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

# Small search to keep this step manageable
parameter_grid = {
    "n_estimators": [100, 150],
    "learning_rate": [0.03, 0.05],
    "max_depth": [2, 3],
}

results = []
best_f1 = -1
best_model = None
best_parameters = None

print(f"\nTraining records: {len(train_df)}")
print(f"Validation records: {len(valid_df)}")
print("\nTesting parameter combinations...")

for parameters in ParameterGrid(parameter_grid):
    model = make_pipeline(parameters)
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    scores = {
        "Accuracy": accuracy_score(y_valid, predictions),
        "Precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_valid, predictions, zero_division=0
        ),
    }

    results.append({
        **parameters,
        **scores,
    })

    print(
        f"Parameters: {parameters} | "
        f"F1: {scores['F1']:.4f}"
    )

    if scores["F1"] > best_f1:
        best_f1 = scores["F1"]
        best_model = model
        best_parameters = parameters

results_df = pd.DataFrame(results).sort_values(
    "F1", ascending=False
)

print("\nTUNING RESULTS — SORTED BY F1")
print(results_df.round(4).to_string(index=False))

print("\nBest parameters:", best_parameters)
print(f"Best validation F1: {best_f1:.4f}")

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

best_model_path = "models/gradient_boosting_tuned.joblib"

import joblib
joblib.dump(best_model, best_model_path)

results_df.to_csv(
    "outputs/gradient_boosting_tuning_results.csv",
    index=False,
)

print(f"\nBest model saved to {best_model_path}")
print("Results saved to outputs/gradient_boosting_tuning_results.csv")
print("\nStep 14 completed successfully.")
