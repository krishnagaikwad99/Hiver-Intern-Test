"""
Evaluation metrics calculation module.
Computes Accuracy, Macro F1, Weighted F1, Per-class Precision/Recall/F1, and Confusion Matrix.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None
) -> Dict[str, Any]:
    """
    Calculates comprehensive classification metrics.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class_metrics = {}
    for label, p, r, f1, sup in zip(labels, per_class_p, per_class_r, per_class_f1, per_class_sup):
        per_class_metrics[label] = {
            "precision": float(round(p, 4)),
            "recall": float(round(r, 4)),
            "f1": float(round(f1, 4)),
            "support": int(sup)
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return {
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "macro_precision": float(round(macro_p, 4)),
        "macro_recall": float(round(macro_r, 4)),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm_df.to_dict()
    }
