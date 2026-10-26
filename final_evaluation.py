
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

print("=" * 60)
print("STEP 18: FINAL TEST EVALUATION")
print("=" * 60)

# Load the tuned model and reserved test data
model = joblib.load(
    "models/gradient_boosting_tuned.joblib"
)

test_df = pd.read_csv(
    "data/processed/classification_test.csv"
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

X_test = test_df[features]
y_test = test_df["DemandClass"]

# Evaluate the model once on the reserved test set
predictions = model.predict(X_test)

results = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, predictions, zero_division=0
    ),
}

print(f"\nTest records: {len(test_df):,}")
print("\nFINAL TEST METRICS")

for metric, value in results.items():
    print(f"{metric}: {value:.4f} ({value * 100:.2f}%)")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal Demand", "High Demand"],
        zero_division=0,
    )
)

# Confusion matrix
cm = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

print("CONFUSION MATRIX")
print(cm)

Path("outputs").mkdir(exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal Demand", "High Demand"],
)

display.plot()
plt.title("Gradient Boosting — Final Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "outputs/final_confusion_matrix.png",
    dpi=300,
)
plt.close()

# Save metrics for the project report
pd.DataFrame([results]).to_csv(
    "outputs/final_classification_metrics.csv",
    index=False,
)

print("\nSaved outputs/final_confusion_matrix.png")
print("Saved outputs/final_classification_metrics.csv")
print("\nStep 18 completed successfully.")
