from sklearn.naive_bayes import GaussianNB

def build_model():
    """
    Returns a Gaussian Naive Bayes classifier.
    Suitable for continuous features in Wine Quality dataset.
    """
    return GaussianNB()
