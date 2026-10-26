
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="EV Charging Demand Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EV Charging Demand Prediction")
st.caption(
    "Explore charging sessions, demand patterns, and machine-learning predictions."
)

DATA_PATH = Path("data/charging_records_clean.csv")
MODEL_PATH = Path("models/gradient_boosting_regression_final.joblib")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

if "StartDatetime" in df.columns:
    df["StartDatetime"] = pd.to_datetime(
        df["StartDatetime"], errors="coerce"
    )

# Sidebar filters
st.sidebar.header("Dashboard Filters")

if "Location" in df.columns:
    locations = sorted(df["Location"].dropna().unique().tolist())
    selected_locations = st.sidebar.multiselect(
        "Select location",
        locations,
        default=locations,
    )
    df = df[df["Location"].isin(selected_locations)]

if "ChargerType" in df.columns:
    charger_types = sorted(df["ChargerType"].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect(
        "Select charger type",
        charger_types,
        default=charger_types,
    )
    df = df[df["ChargerType"].isin(selected_types)]

if df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Charging Sessions", f"{len(df):,}")
col2.metric("Average Demand", f"{df['Demand'].mean():.2f}")
col3.metric("Median Demand", f"{df['Demand'].median():.2f}")
col4.metric("Maximum Demand", f"{df['Demand'].max():.2f}")

st.divider()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("Demand Distribution")
    fig = px.histogram(
        df,
        x="Demand",
        nbins=30,
        title="Distribution of Charging Demand",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Average Demand by Location")
    location_demand = (
        df.groupby("Location", as_index=False)["Demand"]
        .mean()
        .sort_values("Demand", ascending=False)
    )
    fig = px.bar(
        location_demand,
        x="Location",
        y="Demand",
        title="Average Demand by Location",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Charging Sessions by Hour")

if "StartDatetime" in df.columns:
    hourly = df.dropna(subset=["StartDatetime"]).copy()
    hourly["StartHour"] = hourly["StartDatetime"].dt.hour
    hourly_counts = (
        hourly.groupby("StartHour")
        .size()
        .reindex(range(24), fill_value=0)
        .rename("Sessions")
        .reset_index()
    )
    fig = px.line(
        hourly_counts,
        x="StartHour",
        y="Sessions",
        markers=True,
        title="Number of Sessions by Hour",
    )
    st.plotly_chart(fig, use_container_width=True)

# Regression prediction
st.divider()
st.header("Predict Charging Demand")

if not MODEL_PATH.exists():
    st.warning(
        "The final regression model was not found. "
        "Complete Step 19 and check the models folder."
    )
else:
    model = joblib.load(MODEL_PATH)

    st.write("Enter the details of a charging session.")

    with st.form("prediction_form"):
        location_options = sorted(
            pd.read_csv(DATA_PATH)["Location"].dropna().unique().tolist()
        )
        company_options = sorted(
            pd.read_csv(DATA_PATH)["ChargerCompany"].dropna().unique().tolist()
        )
        type_options = sorted(
            pd.read_csv(DATA_PATH)["ChargerType"].dropna().unique().tolist()
        )

        location = st.selectbox("Location", location_options)
        charger_type = st.selectbox("Charger type", type_options)
        company = st.selectbox("Charger company", company_options)

        start_hour = st.slider("Start hour", 0, 23, 18)
        start_day_of_week = st.selectbox(
            "Day of week (Monday = 0)", list(range(7))
        )
        start_month = st.selectbox("Month", list(range(1, 13)))
        start_day_of_month = st.slider("Day of month", 1, 31, 15)
        is_weekend = st.selectbox("Is it a weekend?", [0, 1])

        submitted = st.form_submit_button("Predict Demand")

    if submitted:
        input_data = pd.DataFrame([{
            "Location": location,
            "ChargerType": charger_type,
            "ChargerCompany": company,
            "StartHour": start_hour,
            "StartDayOfWeek": start_day_of_week,
            "StartMonth": start_month,
            "StartDayOfMonth": start_day_of_month,
            "IsWeekend": is_weekend,
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted demand: {prediction:.2f}")
        st.caption(
            "This is a model estimate, not a guaranteed real-world demand value."
        )

# High vs Normal Demand Classification
st.divider()
st.header("🔎 Predict Demand Category")

CLASSIFICATION_MODEL_PATH = Path(
    "models/gradient_boosting_tuned.joblib"
)

if not CLASSIFICATION_MODEL_PATH.exists():
    st.warning(
        "Classification model not found. "
        "Please check that models/gradient_boosting_tuned.joblib exists."
    )
else:
    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    st.write(
        "Enter the charging-session details to predict "
        "whether demand will be High or Normal."
    )

    with st.form("classification_form"):
        class_location = st.selectbox(
            "Classification location",
            sorted(
                pd.read_csv(DATA_PATH)["Location"]
                .dropna().unique().tolist()
            ),
            key="class_location",
        )

        class_type = st.selectbox(
            "Classification charger type",
            sorted(
                pd.read_csv(DATA_PATH)["ChargerType"]
                .dropna().unique().tolist()
            ),
            key="class_type",
        )

        class_company = st.selectbox(
            "Classification charger company",
            sorted(
                pd.read_csv(DATA_PATH)["ChargerCompany"]
                .dropna().unique().tolist()
            ),
            key="class_company",
        )

        class_hour = st.slider(
            "Classification start hour",
            0, 23, 18,
            key="class_hour",
        )

        class_weekday = st.selectbox(
            "Classification day of week (Monday = 0)",
            list(range(7)),
            key="class_weekday",
        )

        class_month = st.selectbox(
            "Classification month",
            list(range(1, 13)),
            key="class_month",
        )

        class_day = st.slider(
            "Classification day of month",
            1, 31, 15,
            key="class_day",
        )

        class_weekend = st.selectbox(
            "Classification weekend? (1 = Yes, 0 = No)",
            [0, 1],
            key="class_weekend",
        )

        classify_submitted = st.form_submit_button(
            "Predict Demand Category"
        )

    if classify_submitted:
        classification_input = pd.DataFrame([{
            "Location": class_location,
            "ChargerType": class_type,
            "ChargerCompany": class_company,
            "StartHour": class_hour,
            "StartDayOfWeek": class_weekday,
            "StartMonth": class_month,
            "StartDayOfMonth": class_day,
            "IsWeekend": class_weekend,
        }])

        predicted_class = int(
            classification_model.predict(classification_input)[0]
        )

        if predicted_class == 1:
            st.success("Predicted Category: HIGH DEMAND")
        else:
            st.info("Predicted Category: NORMAL DEMAND")

        st.caption(
            "High Demand is defined as Demand greater than or equal "
            "to the training-set median threshold of 14.20. "
            "The prediction is an estimate, not a guarantee."
        )

# Model Comparison
st.divider()
st.header("📊 Machine Learning Model Comparison")

st.write(
    "The following charts compare models using validation data. "
    "Final test-set results are displayed separately below."
)

# Regression model comparison: validation results
regression_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    ],
    "MAE": [10.2829, 9.8284, 9.7802, 9.8436],
    "RMSE": [13.4323, 13.0768, 12.9523, 12.8708],
    "R2": [0.0424, 0.0924, 0.1096, 0.1208],
})

st.subheader("Regression Models — Validation Results")

reg_left, reg_right = st.columns(2)

with reg_left:
    fig = px.bar(
        regression_comparison.sort_values("MAE"),
        x="Model",
        y="MAE",
        title="Mean Absolute Error (Lower is Better)",
    )
    st.plotly_chart(fig, use_container_width=True)

with reg_right:
    fig = px.bar(
        regression_comparison.sort_values("RMSE"),
        x="Model",
        y="RMSE",
        title="Root Mean Squared Error (Lower is Better)",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Regression R² Score (Higher is Better)")

fig = px.bar(
    regression_comparison.sort_values("R2", ascending=False),
    x="Model",
    y="R2",
    title="R² Comparison",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View regression comparison table"):
    st.dataframe(
        regression_comparison.round(4),
        use_container_width=True,
        hide_index=True,
    )

# Classification model comparison: validation results
classification_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Tuned Gradient Boosting",
    ],
    "Accuracy": [0.6171, 0.6470, 0.6577, 0.6514, 0.6505],
    "F1 Score": [0.6433, 0.6561, 0.6750, 0.6810, 0.6841],
})

st.subheader("Classification Models — Validation Results")

fig = px.bar(
    classification_comparison.sort_values("F1 Score"),
    x="Model",
    y="F1 Score",
    title="F1 Score Comparison (Higher is Better)",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View classification comparison table"):
    st.dataframe(
        classification_comparison.round(4),
        use_container_width=True,
        hide_index=True,
    )

# Final test results: keep separate from validation results
st.subheader("Final Test-Set Results")

st.caption(
    "These are the final test results from your saved evaluation files. "
    "They are separate from the validation comparisons above."
)

test_col1, test_col2 = st.columns(2)

with test_col1:
    st.markdown("**Regression — Gradient Boosting**")
    st.metric("MAE", "10.1046")
    st.metric("RMSE", "13.1832")
    st.metric("R²", "0.1024")

with test_col2:
    st.markdown("**Classification — Tuned Gradient Boosting**")
    st.metric("Accuracy", "62.66%")
    st.metric("F1 Score", "64.98%")
    st.metric("Recall", "71.04%")

# Feature Importance and Explainability
st.divider()
st.header("🔍 Understanding Model Predictions")

st.subheader("1. Feature Importance")

importance_path = Path("outputs/feature_importance.csv")

if importance_path.exists():
    importance_df = pd.read_csv(importance_path)

    # Identify the feature-name column and its importance values.
    text_columns = importance_df.select_dtypes(
        exclude="number"
    ).columns.tolist()
    numeric_columns = importance_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if text_columns and numeric_columns:
        feature_col = text_columns[0]
        importance_col = numeric_columns[0]

        chart_df = importance_df.sort_values(
            importance_col, ascending=False
        ).head(15)

        fig = px.bar(
            chart_df.sort_values(importance_col),
            x=importance_col,
            y=feature_col,
            orientation="h",
            title="Top 15 Important Features",
            labels={
                importance_col: "Importance",
                feature_col: "Feature",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View feature importance data"):
            st.dataframe(importance_df, use_container_width=True)
    else:
        st.warning(
            "Could not identify feature and importance columns "
            "in feature_importance.csv."
        )
else:
    st.warning("outputs/feature_importance.csv was not found.")

st.subheader("2. SHAP Explainability")

st.write(
    "SHAP helps explain how input features influence model predictions. "
    "The summary plot shows which features have the greatest influence "
    "on the model."
)

shap_plot_path = Path("outputs/shap_summary.png")
shap_data_path = Path("outputs/shap_importance.csv")

if shap_plot_path.exists():
    st.image(
        str(shap_plot_path),
        caption="SHAP Summary Plot",
        use_container_width=True,
    )
else:
    st.info("SHAP summary image was not found.")

if shap_data_path.exists():
    shap_df = pd.read_csv(shap_data_path)

    with st.expander("View SHAP importance data"):
        st.dataframe(shap_df, use_container_width=True)

st.subheader("3. Local Explanation Using LIME")

st.write(
    "LIME explains an individual prediction by showing which features "
    "contributed to that particular result."
)

lime_plot_path = Path("outputs/lime_explanation.png")

if lime_plot_path.exists():
    st.image(
        str(lime_plot_path),
        caption="LIME Explanation for One Example",
        use_container_width=True,
    )
else:
    st.info("LIME explanation image was not found.")

lime_html_path = Path("outputs/lime_explanation.html")

if lime_html_path.exists():
    st.caption(
        "A detailed interactive LIME explanation is also saved at "
        "outputs/lime_explanation.html."
    )

# Data preview
with st.expander("View filtered dataset"):
    st.dataframe(df.head(100), use_container_width=True)

st.caption(
    "Academic project demonstration. Predictions are estimates and should "
    "be interpreted alongside the model evaluation metrics."
)
