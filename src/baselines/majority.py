"""
Baseline 1: Trivial Majority Class Classifier.
Predicts the most frequent class seen in the training data for all inputs.
"""

from typing import List, Dict, Any
from collections import Counter

class MajorityClassClassifier:
    def __init__(self):
        self.majority_class: str = "other_general_query"
        self.classes_: List[str] = []

    def fit(self, X: List[str], y: List[str]):
        """Fits majority class on training labels."""
        if not y:
            return self
        counter = Counter(y)
        self.majority_class = counter.most_common(1)[0][0]
        self.classes_ = list(counter.keys())
        return self

    def predict(self, X: List[str]) -> List[str]:
        """Returns majority class for all inputs."""
        return [self.majority_class] * len(X)

    def predict_proba(self, X: List[str]) -> List[Dict[str, float]]:
        """Returns dummy probability distribution."""
        proba_dict = {c: (1.0 if c == self.majority_class else 0.0) for c in self.classes_}
        return [proba_dict for _ in X]
