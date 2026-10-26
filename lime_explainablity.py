
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

print("=" * 60)
print("STEP 17: LIME EXPLAINABILITY")
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

X_train = train_df[features].copy()

# Convert categorical columns to integer codes for LIME
categorical_columns = [
    "Location",
    "ChargerType",
    "ChargerCompany",
]

category_maps = {}
categorical_indices = []

for column in categorical_columns:
    X_train[column] = X_train[column].astype("category")
    category_maps[column] = list(X_train[column].cat.categories)
    X_train[column] = X_train[column].cat.codes
    categorical_indices.append(features.index(column))

X_train_array = X_train.to_numpy(dtype=float)

def predict_from_lime(data):
    # Convert LIME's numeric input back to the original features
    data_df = pd.DataFrame(data, columns=features)

    for column in categorical_columns:
        categories = category_maps[column]
        codes = data_df[column].round().astype(int)
        codes = codes.clip(0, len(categories) - 1)
        data_df[column] = [
            categories[code] for code in codes
        ]

    data_df["ChargerType"] = data_df["ChargerType"].astype(
        train_df["ChargerType"].dtype
    )
    data_df["ChargerCompany"] = data_df[
        "ChargerCompany"
    ].astype(train_df["ChargerCompany"].dtype)

    # The model expects the original feature names and categories
    probabilities = model.predict_proba(data_df[features])
    return probabilities

# Explain one training record
row_index = 0
instance = X_train_array[row_index]

explainer = LimeTabularExplainer(
    training_data=X_train_array,
    feature_names=features,
    class_names=["Normal Demand", "High Demand"],
    categorical_features=categorical_indices,
    categorical_names={
        features.index(column): [
            str(category) for category in category_maps[column]
        ]
        for column in categorical_columns
    },
    mode="classification",
    random_state=42,
)

print("\nGenerating explanation for one training record...")
explanation = explainer.explain_instance(
    instance,
    predict_from_lime,
    num_features=8,
    num_samples=1000,
)

print("\nLIME explanation for the selected record:")
for feature, weight in explanation.as_list(label=1):
    print(f"{feature}: {weight:.4f}")

Path("outputs").mkdir(exist_ok=True)

explanation.save_to_file("outputs/lime_explanation.html")

fig = explanation.as_pyplot_figure(label=1)
fig.tight_layout()
fig.savefig(
    "outputs/lime_explanation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("\nSaved outputs/lime_explanation.html")
print("Saved outputs/lime_explanation.png")
print("\nStep 17 completed successfully.")
