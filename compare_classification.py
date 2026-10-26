
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

print("=" * 60)
print("STEP 12: CLASSIFICATION MODEL COMPARISON")
print("=" * 60)

# Load training data only; keep test data untouched
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

target = "DemandClass"

# The CSV was created in chronological order.
# Split the earlier 80% for training and later 20% for validation.
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
valid_df = df.iloc[split_index:].copy()

X_train = train_df[features]
y_train = train_df[target]

X_valid = valid_df[features]
y_valid = valid_df[target]

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

def make_pipeline(classifier):
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

results = []

print(f"\nTraining records: {len(train_df)}")
print(f"Validation records: {len(valid_df)}")
print("\nTraining models...")

for name, classifier in models.items():
    print(f"\nTraining {name}...")

    model = make_pipeline(classifier)
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    results.append({
        "Model": name,
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
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1", ascending=False
)

print("\nVALIDATION RESULTS — SORTED BY F1")
print(results_df.round(4).to_string(index=False))

Path("outputs").mkdir(exist_ok=True)
results_df.to_csv(
    "outputs/classification_model_comparison.csv",
    index=False,
)

print("\nResults saved to outputs/classification_model_comparison.csv")
print("\nStep 12 completed successfully.")
