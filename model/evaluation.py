import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


def evaluate_multiclass(y_true, y_pred, y_proba, classes):
    """
    Assignment 2 metrics:
    Accuracy, AUC (multiclass OVR), Precision, Recall, F1, MCC
    """
    metrics = {}

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["Recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["F1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=classes)
    metrics["AUC"] = roc_auc_score(
        y_true_bin, y_proba, multi_class="ovr", average="macro"
    )

    return metrics
