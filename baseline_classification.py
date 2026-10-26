
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

print("=" * 60)
print("STEP 11: BASELINE CLASSIFICATION")
print("=" * 60)

# Load the prepared datasets
train_df = pd.read_csv(
    "data/processed/classification_train.csv"
)
test_df = pd.read_csv(
    "data/processed/classification_test.csv"
)

# Features available before a charging session starts
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

target = "DemandClass"

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

# Encode categories and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
    ]
)

# Baseline: always predict the most common training class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

# Logistic Regression pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Compare baseline and Logistic Regression
results = []

for name, predicted in [
    ("Dummy Baseline", baseline_predictions),
    ("Logistic Regression", predictions),
]:
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(
            y_test, predicted, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predicted, zero_division=0
        ),
        "F1": f1_score(
            y_test, predicted, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)

print("\nTEST RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nLOGISTIC REGRESSION CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal Demand", "High Demand"],
        zero_division=0,
    )
)

print("CONFUSION MATRIX")
print(
    confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )
)

# Save model and results
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/logistic_regression_baseline.joblib",
)

results_df.to_csv(
    "outputs/baseline_classification_results.csv",
    index=False,
)

print("\nModel saved to models/logistic_regression_baseline.joblib")
print("Results saved to outputs/baseline_classification_results.csv")
print("\nStep 11 completed successfully.")
