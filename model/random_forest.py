from sklearn.ensemble import RandomForestClassifier

def build_model():
    """
    Returns a Random Forest classifier.
    Good at handling non-linear relationships and feature interactions.
    """
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
