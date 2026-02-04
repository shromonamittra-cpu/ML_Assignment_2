Multi-Model Machine Learning Classifier – Wine Quality Dataset
a. Problem Statement

The objective of this project is to design, evaluate, and deploy multiple supervised machine learning classification models to predict wine quality scores using physicochemical properties of red and white wines. The project demonstrates an end-to-end machine learning workflow including data ingestion, model training, comparative evaluation, and deployment through an interactive Streamlit web application.

b. Dataset Description

Dataset Name: Wine Quality Dataset (Red + White)
Source: UCI Machine Learning Repository

The dataset consists of physicochemical measurements of red and white wines, with the task of predicting discrete wine quality scores.

Target Variable:

quality → Multiclass classification (integer wine quality score)

Dataset Characteristics:

Total samples: ~6,497

Total features: 11 physicochemical attributes

Additional feature: wine_type (0 = Red, 1 = White)

Problem type: Multiclass classification

c. Models Used and Evaluation Metrics

The following six classification models were implemented and evaluated on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

Each model was evaluated using the following metrics:

Accuracy

Area Under Curve (AUC)

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

Model Performance Comparison
Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Random Forest	0.695385	0.800662	0.541693	0.367415	0.400948	0.530974
XGBoost	0.636923	0.835929	0.468721	0.342482	0.374957	0.440217
Decision Tree	0.595385	0.623743	0.326788	0.333161	0.328894	0.398883
Logistic Regression	0.540769	0.792099	0.377844	0.227959	0.233023	0.270679
KNN	0.543846	0.680549	0.224089	0.233059	0.227768	0.291602
Naive Bayes (Gaussian)	0.364615	0.711231	0.239126	0.378991	0.224629	0.113606
d. Model Performance Observations
Model	Observation
Random Forest	Achieved the most balanced performance across accuracy, F1-score, and MCC, indicating strong generalization capability for multiclass wine quality prediction.
XGBoost	Delivered competitive performance with the highest AUC, demonstrating effective handling of non-linear feature interactions and class imbalance.
Decision Tree	Captured non-linear patterns in the data but showed signs of overfitting, resulting in less stable performance compared to ensemble models.
Logistic Regression	Served as a reliable baseline model but struggled with complex decision boundaries and imbalanced quality labels.
KNN	Performance was sensitive to neighborhood selection and feature scaling, leading to moderate accuracy and inconsistent recall across classes.
Naive Bayes (Gaussian)	Showed higher recall but lower precision and MCC due to strong independence assumptions among features.
e. Streamlit Application Features

A Streamlit-based interactive web application was developed and deployed using Streamlit Community Cloud, providing the following features:

Dataset preview

Model selection via dropdown menu

Automatic model training and evaluation

Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC)

Confusion matrix and classification report visualization

Downloadable test dataset for user experimentation

f. Conclusion

This project demonstrates a complete machine learning lifecycle, from data preprocessing and model comparison to web-based deployment. Ensemble methods such as Random Forest and XGBoost outperform single models in handling the complexity and imbalance inherent in multiclass wine quality prediction. The modular design and interactive Streamlit interface make the system extensible, interpretable, and suitable for comparative model analysis in real-world classification tasks.