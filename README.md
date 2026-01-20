Machine Learning Assignment 2

Classification Models with Streamlit Deployment

Project Overview

This project implements multiple machine learning classification models on the UCI Wine Quality dataset (Red + White) and deploys the trained models using Streamlit.
The application allows users to select a classification model, trains it on the dataset, and displays evaluation metrics and a confusion matrix interactively.

The goal of this assignment is to:

Apply supervised classification techniques

Compare multiple models

Evaluate performance using standard metrics

Deploy the solution as a web application

Dataset Description

Dataset: UCI Wine Quality Dataset (Red + White combined)

Source: UCI Machine Learning Repository

Samples: ~6,497 wine samples

Features: 11 physicochemical attributes

Target variable: quality (integer score)

Key Features Include:

Fixed acidity

Volatile acidity

Citric acid

Residual sugar

Chlorides

Free sulfur dioxide

Total sulfur dioxide

Density

pH

Sulphates

Alcohol

An additional feature wine_type is added:

0 → Red wine

1 → White wine

Machine Learning Models Implemented

The following classification models are implemented in a modular structure:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes

Random Forest

XGBoost (optional, if supported)

Each model is implemented in a separate file under the model/ directory.

Evaluation Metrics

The models are evaluated using the following metrics:

Accuracy – Overall correctness of predictions

AUC (Area Under Curve) – Multiclass One-Vs-Rest ROC AUC

Precision – Correct positive predictions

Recall – Ability to identify all relevant classes

F1-Score – Harmonic mean of precision and recall

MCC (Matthews Correlation Coefficient) – Balanced performance metric for multiclass classification

A confusion matrix and classification report are also displayed.

Streamlit Web Application

The Streamlit app provides:

Dataset preview

Model selection dropdown

Automatic model training

Real-time evaluation metrics

Confusion matrix visualization

Classification report

The app is deployed using Streamlit Cloud.

Live Deployment

The application is deployed on Streamlit Cloud and can be accessed via the provided Streamlit app URL.

Observations

Logistic Regression provides a baseline performance.

Tree-based models (Decision Tree, Random Forest) generally improve classification results.

Random Forest shows better balance across precision, recall, and MCC.

Multiclass classification is challenging due to class imbalance in wine quality scores.

Conclusion

This project demonstrates the complete machine learning workflow:

Data preprocessing

Model building

Performance evaluation

Web-based deployment

The modular design allows easy extension with additional models or datasets.

Author

Name: Sramana Mittra 
Course: Machine Learning
Assignment: ML Assignment 2
