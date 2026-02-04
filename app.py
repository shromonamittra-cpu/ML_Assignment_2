import streamlit as st
import pandas as pd
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
# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Wine Quality ML Classifier",
    layout="centered"
)
# --------------------------------------------------
# Title and description
# --------------------------------------------------
st.title("Wine Quality Classification App")
st.caption(
    "An interactive comparison of multiple supervised machine learning models for "
    "multiclass wine quality prediction using physicochemical attributes."
)
st.markdown("---")
# --------------------------------------------------
# Download test dataset
# --------------------------------------------------
st.subheader("Download Dataset")
TEST_CSV_URL = "https://raw.githubusercontent.com/shromonamittra-cpu/ML_Assignment_2/main/wine_quality_test.csv"
st.markdown(f"[📄 View raw CSV]({TEST_CSV_URL})")
try:
    test_df = pd.read_csv(TEST_CSV_URL)
    csv_bytes = test_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download wine_quality_test.csv",
        data=csv_bytes,
        file_name="wine_quality_test.csv",
        mime="text/csv",
    )
except Exception:
    st.warning("Download button unavailable. Use the link above.")

df = load_wine_quality()

with st.expander("📊 Dataset Preview"):
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

X = df.drop(columns=["quality"])
y = df["quality"].astype(int)

classes = np.sort(y.unique())
num_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Model selection
# ===============================
MODEL_PATHS = {
    "Logistic Regression": "artifacts/logistic.pkl",
    "Decision Tree": "artifacts/decision_tree.pkl",
    "KNN": "artifacts/knn.pkl",
    "Naive Bayes": "artifacts/naive_bayes.pkl",
    "Random Forest": "artifacts/random_forest.pkl",
}

if XGB_AVAILABLE:
    MODEL_PATHS["XGBoost"] = "artifacts/xgboost.pkl"

model_name = st.selectbox("Select Model", list(MODEL_PATHS.keys()))

# ===============================
# Load pre-trained model (NO TRAINING)
# ===============================
try:
    model = joblib.load(MODEL_PATHS[model_name])
except Exception as e:
    st.error("❌ Failed to load pre-trained model")
    st.error(str(e))
    st.stop()

# ===============================
# Predict + Evaluate
# ===============================
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
