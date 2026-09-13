"""
Baseline 2: TF-IDF + Logistic Regression Classifier.
Uses word n-gram TF-IDF features and a Logistic Regression model with balanced class weights.
"""

from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TFIDFBaselineClassifier:
    def __init__(self, max_features: int = 1000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
        self.clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        self.is_fitted = False
        self.classes_ = []

    def fit(self, X: List[str], y: List[str]):
        """Fits TF-IDF vectorizer and Logistic Regression classifier."""
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        self.classes_ = list(self.clf.classes_)
        self.is_fitted = True
        return self

    def predict(self, X: List[str]) -> List[str]:
        """Predicts intent labels for given customer messages."""
        if not self.is_fitted:
            raise ValueError("Classifier is not fitted yet.")
        X_vec = self.vectorizer.transform(X)
        return list(self.clf.predict(X_vec))

    def predict_proba(self, X: List[str]) -> List[Dict[str, float]]:
        """Returns dict of intent probabilities per sample."""
        if not self.is_fitted:
            raise ValueError("Classifier is not fitted yet.")
        X_vec = self.vectorizer.transform(X)
        probas = self.clf.predict_proba(X_vec)
        results = []
        for prob in probas:
            results.append({cls: float(p) for cls, p in zip(self.classes_, prob)})
        return results
