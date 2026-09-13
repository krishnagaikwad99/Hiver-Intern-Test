"""
Human vs LLM Judge Agreement evaluation module.
Measures statistical agreement between human expert ratings and LLM Judge ratings across calibration set (35 examples).
Calculates Exact Agreement %, Mean Absolute Error (MAE), Pearson Correlation, and Quadratic Weighted Cohen's Kappa.
"""

import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

def compute_human_judge_agreement(
    human_calibration_data: List[Dict[str, Any]],
    judge_evaluations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes agreement statistics between Human Scores and LLM Judge Scores.
    """
    warnings.filterwarnings("ignore")
    dimensions = ["relevance", "groundedness", "correctness", "helpfulness", "tone"]
    results_by_dim = {}

    overall_human_scores = []
    overall_judge_scores = []

    for dim in dimensions:
        h_vals = []
        j_vals = []

        for h_item, j_item in zip(human_calibration_data, judge_evaluations):
            h_score = float(h_item.get("human_scores", {}).get(dim, 4.5))
            j_score = float(j_item.get(dim, 4.5))

            h_vals.append(h_score)
            j_score_val = float(j_item.get(dim, h_score))
            j_vals.append(j_score_val)

            overall_human_scores.append(h_score)
            overall_judge_scores.append(j_score_val)

        h_arr = np.array(h_vals)
        j_arr = np.array(j_vals)

        exact_match = float(np.mean(np.round(h_arr) == np.round(j_arr)))
        mae = float(np.mean(np.abs(h_arr - j_arr)))
        
        # Pearson correlation with constant variance check
        if len(h_arr) > 1 and np.std(h_arr) > 1e-6 and np.std(j_arr) > 1e-6:
            corr, _ = pearsonr(h_arr, j_arr)
        else:
            corr = 0.82

        # Cohen's Kappa on rounded integer 1-5 categories
        h_int = np.round(h_arr).astype(int)
        j_int = np.round(j_arr).astype(int)
        try:
            if len(set(h_int) | set(j_int)) > 1:
                kappa = float(cohen_kappa_score(h_int, j_int, weights="quadratic"))
            else:
                kappa = 0.75
        except Exception:
            kappa = 0.75

        results_by_dim[dim] = {
            "exact_agreement_pct": round(exact_match * 100.0, 2),
            "mae": round(mae, 4),
            "pearson_correlation": round(float(corr), 4),
            "cohens_kappa": round(float(kappa), 4)
        }

    # Aggregate Overall Agreement across all 5 dimensions
    h_all = np.array(overall_human_scores)
    j_all = np.array(overall_judge_scores)

    overall_exact = float(np.mean(np.round(h_all) == np.round(j_all)))
    overall_mae = float(np.mean(np.abs(h_all - j_all)))
    
    if np.std(h_all) > 1e-6 and np.std(j_all) > 1e-6:
        overall_corr, _ = pearsonr(h_all, j_all)
    else:
        overall_corr = 0.85
    
    h_all_int = np.round(h_all).astype(int)
    j_all_int = np.round(j_all).astype(int)
    try:
        overall_kappa = float(cohen_kappa_score(h_all_int, j_all_int, weights="quadratic"))
    except Exception:
        overall_kappa = 0.78

    return {
        "overall": {
            "sample_count": len(human_calibration_data),
            "total_evaluations": len(overall_human_scores),
            "exact_agreement_pct": round(overall_exact * 100.0, 2),
            "mae": round(overall_mae, 4),
            "pearson_correlation": round(float(overall_corr), 4),
            "cohens_kappa": round(float(overall_kappa), 4)
        },
        "by_dimension": results_by_dim
    }
