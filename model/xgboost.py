from xgboost import XGBClassifier

def build_model(num_classes):
    """
    Returns an XGBoost multiclass classifier.
    """
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42
    )
