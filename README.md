# EV Charging Demand Prediction and Usage Pattern Analysis

## Project Overview

This project uses machine learning to predict electric vehicle (EV) charging demand and analyze charging usage patterns across different locations and time periods.

The project includes data cleaning, exploratory data analysis (EDA), regression, classification, model evaluation, and explainability techniques.

## Objectives

- Analyze EV charging patterns by location, charger type, and time.
- Predict charging demand using regression models.
- Classify demand into High and Normal categories.
- Compare machine learning models using evaluation metrics.
- Explain model predictions using feature importance, SHAP, and LIME.
- Present results through an interactive Streamlit dashboard.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- SHAP
- LIME
- Joblib

## Machine Learning

### Regression

The selected regression model is Gradient Boosting.

Test results:
- MAE: 10.10
- RMSE: 13.18
- R²: 0.1024

### Classification

The classification model predicts whether demand is High or Normal.

Test results:
- Accuracy: 62.66%
- High-demand precision: 59.87%
- High-demand recall: 71.04%
- High-demand F1-score: 64.98%

These are the current test results for this project. Performance may vary with new data.

## Explainability

The project uses:
- Feature importance
- SHAP
- LIME

StartHour was the most important feature in the trained classification model.

## Run the Dashboard

1. Install Python.
2. Open a terminal in the project folder.
3. Install the required packages using `requirements.txt`.
4. Ensure the required dataset and trained model files are available in their expected locations.
5. Run:

```bash
python -m streamlit run app.py
```

## Data and Model Files

Raw and cleaned datasets and saved `.joblib` models are excluded from this Git repository. You must supply the required files locally before running the complete dashboard.

## Project Structure

- `app.py` — Streamlit dashboard
- `clean_data.py` — Data cleaning
- `eda.py` — Exploratory data analysis
- `baseline_regression.py` — Regression baseline
- `baseline_classification.py` — Classification baseline
- `final_regression.py` — Final regression workflow
- `final_evaluation.py` — Model evaluation
- `feature_importance.py` — Feature importance analysis
- `shap_explainablity.py` — SHAP explanations
- `lime_explainablity.py` — LIME explanations
- `outputs/` — Evaluation results and visualizations

## Limitations

Model predictions depend on the available dataset and selected features. Further data collection, feature engineering, and model tuning may improve performance.
