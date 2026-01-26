**Assignment Details:**
Project Overview:

This project implements multiple machine learning classification models on the UCI Wine Quality dataset (Red + White) and deploys the trained models using Streamlit. The application allows users to select a classification model, train it on the dataset, and display evaluation metrics and a confusion matrix interactively.
Problem Statement:

The objective of this project is to build and evaluate multiple machine learning classification models for predicting wine quality based on physicochemical properties, and to deploy these models through an interactive Streamlit web application for real-time evaluation and comparison.
Dataset Description:

The dataset used in this project is the UCI Wine Quality dataset (red and white wines combined), consisting of approximately 6,497 samples with 11 physicochemical features used to predict discrete wine quality scores.
Dataset: UCI Wine Quality Dataset (Red + White combined)
Source: UCI Machine Learning Repository
Samples: ~6,497 wine samples
Features: 11 physicochemical attributes
Target variable: quality (integer score)
Key features:
•	Fixed acidity
•	Volatile acidity
•	Citric acid
•	Residual sugar
•	Chlorides
•	Free sulfur dioxide
•	Total sulfur dioxide
•	Density
•	pH
•	Sulphates
•	Alcohol
•	An additional feature wine_type is added: 0 → Red wine and 1 → White wine
Machine Learning Models Implemented:
The following classification models are implemented in a modular structure:
•	Logistic Regression
•	Decision Tree Classifier
•	K-Nearest Neighbors (KNN)
•	Naive Bayes
•	Random Forest
•	XGBoost 
Model Performance Comparison Analysis:

Model
	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Random Forest	0.695385	0.800662	0.541693	0.367415	0.400948	0.530974
XGBoost	0.636923	0.835929	0.468721	0.342482	0.374957	0.440217
Decision Tree	0.595385	0.623743	0.326788	0.333161	0.328894	0.398883
Logistic Regression	0.540769	0.792099	0.377844	0.227959	0.233023	0.270679
kNN	0.543846	0.680549	0.224089	0.233059	0.227768	0.291602
Naive Bayes (Gaussian)	0.364615	0.711231	0.239126	0.378991	0.224629	0.113606

Model-wise Observations:

Model
	Observations
Random Forest	Random Forest achieved the most balanced performance across accuracy, F1-score, and MCC, indicating strong generalization on the multiclass wine quality data.
XGBoost	XGBoost delivered competitive performance with improved AUC, demonstrating effective handling of non-linear feature interactions and class imbalance.
Decision Tree	Decision Tree captured non-linear patterns in the data but showed signs of overfitting, leading to less stable performance compared to ensemble methods.
Logistic Regression	Logistic Regression provided a reliable baseline model but struggled with complex class boundaries and imbalanced quality labels.
kNN	Performance of kNN was sensitive to neighborhood selection and feature scaling, resulting in moderate accuracy and inconsistent recall across classes.
Naive Bayes (Gaussian)	Gaussian Naive Bayes performed reasonably well despite its independence assumption, but its simplified probabilistic model limited overall predictive accuracy.

Streamlit Web Application:
A Streamlit-based interactive web application was developed to demonstrate and evaluate multiple supervised machine learning classification models using the UCI Wine Quality dataset. The models were implemented, trained, and evaluated in a Jupyter Notebook and subsequently integrated into a Streamlit application for interactive visualization and comparison.
Key Features:
•	End-to-end implementation of multiple classification models in a modular pipeline
•	Automated data loading and preprocessing for red and white wine datasets
•	Train–test split with stratification to preserve class distribution
•	Interactive model selection within the Streamlit interface
•	Display of comprehensive evaluation metrics, including:
o	Accuracy
o	Precision
o	Recall
o	F1-Score
o	ROC-AUC (multiclass, One-vs-Rest)
o	Matthews Correlation Coefficient (MCC)
•	Visualization of confusion matrices for multiclass classification
•	Generation of detailed classification reports
•	Clean, intuitive, and user-friendly interface for model comparison
Models Implemented:
The following machine learning models were implemented and evaluated in the notebook and exposed through the Streamlit application:
•	Logistic Regression
•	Decision Tree Classifier
•	K-Nearest Neighbors (KNN)
•	Gaussian Naive Bayes
•	Random Forest Classifier
•	XGBoost Classifier 
Deployment:
The trained models and evaluation workflow were deployed using Streamlit Community Cloud, enabling real-time interaction, metric visualization, and comparative analysis without requiring local execution.
Project Structure:
ML_Assignment_2/
│
├── app.py
│   ├─ Streamlit application entry point
│   ├─ Loads dataset, handles UI, model selection, training, and evaluation
│
├── requirements.txt
│   └─ Lists all Python dependencies required to run the application
│
├── README.md
│   └─ Project documentation and usage instructions
│
├── wine_quality_test.csv
│   └─ Sample labeled test dataset for external testing and download
│
└── model/
    │
    ├── logistic.py
    │   └─ Logistic Regression model implementation
    │
    ├── decision_tree.py
    │   └─ Decision Tree classifier implementation
    │
    ├── knn.py
    │   └─ K-Nearest Neighbors classifier implementation
    │
    ├── naive_bayes.py
    │   └─ Gaussian Naive Bayes classifier implementation
    │
    ├── random_forest.py
    │   └─ Random Forest classifier implementation
    │
    ├── xgboost.py
    │   └─ XGBoost classifier implementation 
    │
    └── evaluation.py
        └─ Centralized evaluation utilities for computing performance metrics
Conclusion:
This project successfully demonstrates the end-to-end machine learning workflow, from data preprocessing and model training to evaluation and web-based deployment. Multiple classification models were implemented and compared on the Wine Quality dataset, highlighting the strengths and limitations of each approach in a multiclass setting. The Streamlit application enables interactive model selection, performance analysis, and result visualization, making the system accessible and user-friendly. Overall, the project reinforces key concepts of supervised learning, model evaluation, and deployment, while showcasing the importance of modular design and reproducibility in practical machine learning applications.
