import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from model.evaluation import evaluate_multiclass

# ===============================
# Page config + startup render
# ===============================
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.write("✅ App started")
# ===============================
# Paths to pretrained models
# ===============================
MODEL_PATHS = {
    "Logistic Regression": "artifacts/logistic.pkl",
    "Decision Tree": "artifacts/decision_tree.pkl",
    "KNN": "artifacts/knn.pkl",
    "Naive Bayes": "artifacts/naive_bayes.pkl",
    "Random Forest": "artifacts/random_forest.pkl",
    "XGBoost": "artifacts/xgboost.pkl",
}
# ===============================
# Dataset loader
# ===============================
@st.cache_data(show_spinner=False)
def load_wine_quality():
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    df_red = pd.read_csv(red_url, sep=";")
    df_white = pd.read_csv(white_url, sep=";")

    df_red["wine_type"] = 0
    df_white["wine_type"] = 1

    return pd.concat([df_red, df_white], ignore_index=True)

# ===============================
# Load model safely
# ===============================
@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

# ===============================
# UI
# ===============================
st.title("Machine Learning Assignment 2")
st.subheader("Wine Quality Classification (Red + White)")

# ===============================
# Download test dataset
# ===============================
st.subheader("⬇️ Download Test Dataset")

TEST_CSV_URL = "https://raw.githubusercontent.com/shromonamittra-cpu/ML_Assignment_2/main/wine_quality_test.csv"
st.markdown(f"[📄 View raw CSV]({TEST_CSV_URL})")

try:
    test_df = pd.read_csv(TEST_CSV_URL)
    st.download_button(
        label="📥 Download wine_quality_test.csv",
        data=test_df.to_csv(index=False).encode("utf-8"),
        file_name="wine_quality_test.csv",
        mime="text/csv",
    )
except Exception:
    st.warning("Test dataset not available.")

# ===============================
# Load data
# ===============================
df = load_wine_quality()

with st.expander("📊 Dataset Preview"):
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

X = df.drop(columns=["quality"])
y = df["quality"].astype(int)

classes = np.sort(y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Model selection
# ===============================
available_models = [m for m in MODEL_PATHS if MODEL_PATHS[m] is not None]
model_name = st.selectbox("Select Model", available_models)

# ===============================
# Load model (NO TRAINING)
# ===============================
try:
    model = load_model(MODEL_PATHS[model_name])
except Exception as e:
    st.error("❌ Failed to load model file")
    st.error(str(e))
    st.stop()

# ===============================
# Predict + Evaluate
# ===============================
y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)
else:
    st.error("Model does not support probability prediction.")
    st.stop()

metrics = evaluate_multiclass(y_test, y_pred, y_proba, classes)

# ===============================
# Metrics display
# ===============================
st.subheader("📈 Evaluation Metrics")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
c2.metric("AUC", f"{metrics['AUC']:.3f}")
c3.metric("Precision", f"{metrics['Precision']:.3f}")
c4.metric("Recall", f"{metrics['Recall']:.3f}")
c5.metric("F1 Score", f"{metrics['F1']:.3f}")
c6.metric("MCC", f"{metrics['MCC']:.3f}")

# ===============================
# Confusion Matrix
# ===============================
st.subheader("🧮 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_df = pd.DataFrame(
    cm,
    index=[f"True {c}" for c in classes],
    columns=[f"Pred {c}" for c in classes]
)
st.dataframe(cm_df)

# ===============================
# Classification Report
# ===============================
st.subheader("📄 Classification Report")
st.code(classification_report(y_test, y_pred, zero_division=0))

