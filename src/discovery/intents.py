"""
Intent discovery module.
Extracts top TF-IDF keyphrases and clusters customer queries to discover brand-specific intent patterns.
"""

import yaml
import pandas as pd
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer

def load_intent_taxonomy(yaml_path: str = "config/intents.yaml") -> Dict[str, Any]:
    """Loads the intent taxonomy YAML configuration."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_intents_from_queries(queries: List[str], top_n_keywords: int = 15) -> Dict[str, List[str]]:
    """
    Exploratory TF-IDF keyphrase discovery to identify candidate intent topics from customer queries.
    """
    if not queries:
        return {}

    vectorizer = TfidfVectorizer(max_features=50, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(queries)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate top keywords by mean TF-IDF weight
    mean_weights = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_weights.argsort()[::-1][:top_n_keywords]
    
    discovered_keywords = [feature_names[i] for i in top_indices]
    
    return {
        "top_keywords": discovered_keywords,
        "sample_query_count": len(queries)
    }
