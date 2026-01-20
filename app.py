import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# Page config + startup render
# ===============================
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.write("✅ App started")

# ===============================
# Safe model imports
# ===============================
from model.logistic import build_model as build_logistic
from model.decision_tree import build_model as build_dt
from model.knn import build_model as build_knn
from model.naive_bayes import build_model as build_nb
from model.random_forest import build_model as build_rf

# XGBoost is OPTIONAL (Streamlit Cloud safe)
try:
    from model.xgboost import build_model as build_xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from model.evaluation import evaluate_multiclass


# ===============================
# Dataset loader
# ===============================
@st.cache_data(show_spinner=False)
def load_wine_quality():
    try:
        red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

        df_red = pd.read_csv(red_url, sep=";")
        df_white = pd.read_csv(white_url, sep=";")

        df_red["wine_type"] = 0
        df_white["wine_type"] = 1

        return pd.concat([df_red, df_white], ignore_index=True)

    except Exception as e:
        st.error("❌ Failed to load dataset")
        st.error(str(e))
        st.stop()


# ===============================
# UI
# ===============================
st.title("Machine Learning Assignment 2")
st.subheader("Wine Quality Classification (Red + White)")

df = load_wine_quality()

with st.expander("📊 Dataset Preview"):
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

X = df.drop(columns=["quality"])
y = df["quality"].astype(int)

classes = np.sort(y.unique())
num_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Model selection
# ===============================
model_dict = {
    "Logistic Regression": build_logistic(),
    "Decision Tree": build_dt(),
    "KNN": build_knn(),
    "Naive Bayes": build_nb(),
    "Random Forest": build_rf(),
}

if XGB_AVAILABLE:
    model_dict["XGBoost"] = build_xgb(num_classes)

model_name = st.selectbox("Select Model", list(model_dict.keys()))
model = model_dict[model_name]

# ===============================
# Train + Evaluate
# ===============================
with st.spinner("Training model..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)
else:
    st.error("Selected model does not support probability prediction.")
    st.stop()

metrics = evaluate_multiclass(
    y_test,
    y_pred,
    y_proba,
    classes=classes
)

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
