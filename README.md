# ML_Assignment_2  
**Machine Learning Assignment 2: Classification Models with Streamlit Deployment**

---

## Project Overview

This project implements and compares multiple machine learning classification models on the **UCI Wine Quality dataset (Red and White wines combined)** and deploys the solution using a **Streamlit-based interactive web application**.  
The application allows users to select a model, evaluate its performance, and visualize results using standard classification metrics.

---

## Dataset Description

The dataset used in this project is the UCI Wine Quality dataset (red and white wines combined), consisting of approximately 6,497 samples with 11 physicochemical features used to predict discrete wine quality scores.

---

## Machine Learning Models Implemented

The following classification models were implemented in a modular manner:

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Naive Bayes (Gaussian)  
- Random Forest  
- XGBoost (optional, if supported)

Each model is implemented in a separate Python file under the `model/` directory.

---

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy** – Overall correctness of predictions  
- **AUC (Area Under Curve)** – Multiclass One-vs-Rest ROC AUC  
- **Precision** – Correct positive predictions  
- **Recall** – Ability to identify all relevant classes  
- **F1-Score** – Harmonic mean of precision and recall  
- **MCC (Matthews Correlation Coefficient)** – Balanced metric for multiclass classification  

A confusion matrix and classification report are also generated.

---

## Model Performance Comparison Analysis

| Model                  | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|------------------------|----------|---------|-----------|---------|----------|---------|
| **Random Forest**      | 0.695385 | 0.800662 | 0.541693  | 0.367415 | 0.400948 | 0.530974 |
| **XGBoost**            | 0.636923 | 0.835929 | 0.468721  | 0.342482 | 0.374957 | 0.440217 |
| **Decision Tree**      | 0.595385 | 0.623743 | 0.326788  | 0.333161 | 0.328894 | 0.398883 |
| **Logistic Regression**| 0.540769 | 0.792099 | 0.377844  | 0.227959 | 0.233023 | 0.270679 |
| **KNN**                | 0.543846 | 0.680549 | 0.224089  | 0.233059 | 0.227768 | 0.291602 |
| **Naive Bayes (Gaussian)** | 0.364615 | 0.711231 | 0.239126  | 0.378991 | 0.224629 | 0.113606 |


---

## Model-wise Observations

| Model | Observations |
|------|-------------|
| **Random Forest** | Random Forest achieved the most balanced performance across accuracy, F1-score, and MCC, indicating strong generalization on the multiclass wine quality data. |
| **XGBoost** | XGBoost delivered competitive performance with improved AUC, demonstrating effective handling of non-linear feature interactions and class imbalance. |
| **Decision Tree** | Decision Tree captured non-linear patterns in the data but showed signs of overfitting, leading to less stable performance compared to ensemble methods. |
| **Logistic Regression** | Logistic Regression provided a reliable baseline model but struggled with complex class boundaries and imbalanced quality labels. |
| **KNN** | Performance of KNN was sensitive to neighborhood selection and feature scaling, resulting in moderate accuracy and inconsistent recall across classes. |
| **Naive Bayes (Gaussian)** | Gaussian Naive Bayes performed poorly overall due to strong independence assumptions that do not hold well for the wine quality features. |

---