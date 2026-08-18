
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #dddddd;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        "model/preprocessor.pkl"
    )

    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),
        "kNN": joblib.load(
            "model/knn.pkl"
        ),
        "Gaussian Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),
        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }

    return preprocessor, models


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Bank Marketing Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive comparison of five machine learning classification models'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODELS
# ============================================================

try:

    preprocessor, models = load_models()

except Exception as e:

    st.error(
        "Unable to load the saved models. "
        "Please verify the model folder and files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select a classification model",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload the held-out test dataset used during model evaluation."
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload the test CSV file to begin evaluation."
    )

    st.stop()


# ============================================================
# READ DATA
# ============================================================

try:

    uploaded_data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("Unable to read the uploaded CSV file.")
    st.exception(e)
    st.stop()


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

missing_features = [
    column
    for column in required_features
    if column not in uploaded_data.columns
]

if missing_features:

    st.error(
        "The uploaded dataset is missing required columns:"
    )

    st.write(missing_features)

    st.stop()


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

st.subheader("2. Dataset Preview")

st.write(
    f"Rows: **{uploaded_data.shape[0]:,}**  |  "
    f"Columns: **{uploaded_data.shape[1]:,}**"
)

st.dataframe(
    uploaded_data.head(10),
    use_container_width=True
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X_uploaded = uploaded_data[required_features]

try:

    X_uploaded_processed = preprocessor.transform(
        X_uploaded
    )

except Exception as e:

    st.error(
        "The uploaded dataset could not be processed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREDICTIONS
# ============================================================

try:

    predictions = selected_model.predict(
        X_uploaded_processed
    )

    probabilities = selected_model.predict_proba(
        X_uploaded_processed
    )[:, 1]

except Exception as e:

    st.error(
        "Prediction failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREDICTION SUMMARY
# ============================================================

st.subheader("3. Prediction Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Selected Model",
        selected_model_name
    )

with col2:
    st.metric(
        "Predicted No",
        int((predictions == 0).sum())
    )

with col3:
    st.metric(
        "Predicted Yes",
        int((predictions == 1).sum())
    )


# ============================================================
# EVALUATION
# ============================================================

if "y" in uploaded_data.columns:

    y_actual = uploaded_data["y"].map({
        "no": 0,
        "yes": 1
    })

    if y_actual.isnull().any():

        st.warning(
            "The target column contains values other than "
            "'no' and 'yes'. Evaluation cannot be completed."
        )

    else:

        st.subheader("4. Evaluation Metrics")

        accuracy = accuracy_score(
            y_actual,
            predictions
        )

        auc = roc_auc_score(
            y_actual,
            probabilities
        )

        precision = precision_score(
            y_actual,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_actual,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_actual,
            predictions,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_actual,
            predictions
        )

        metric_columns = st.columns(6)

        metric_columns[0].metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        metric_columns[1].metric(
            "AUC",
            f"{auc:.4f}"
        )

        metric_columns[2].metric(
            "Precision",
            f"{precision:.4f}"
        )

        metric_columns[3].metric(
            "Recall",
            f"{recall:.4f}"
        )

        metric_columns[4].metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        metric_columns[5].metric(
            "MCC",
            f"{mcc:.4f}"
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.subheader("5. Confusion Matrix")

        cm = confusion_matrix(
            y_actual,
            predictions
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.subheader("6. Classification Report")

        report = classification_report(
            y_actual,
            predictions,
            target_names=["No", "Yes"],
            digits=4,
            zero_division=0
        )

        st.code(
            report,
            language="text"
        )

else:

    st.info(
        "The uploaded file does not contain the 'y' target column. "
        "Predictions are displayed, but evaluation metrics cannot "
        "be calculated."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "M.Tech AIML — Machine Learning Assignment 2 | "
    "Bank Marketing Classification"
)
