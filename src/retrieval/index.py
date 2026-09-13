"""
Vector retrieval index manager.
Builds, saves, loads, and queries vector indices of historical customer support resolutions.
Supports sentence-transformers or TF-IDF vector representations with cosine similarity lookup.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResolutionVectorStore:
    def __init__(self, embedding_type: str = "tfidf"):
        self.embedding_type = embedding_type
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
        self.vectors: np.ndarray = None
        self.metadata: List[Dict[str, Any]] = []
        self.is_built = False

    def build_index(self, conversations: List[Dict[str, Any]]):
        """
        Builds vector index from reconstructed conversation pairs.
        Each item has customer_message, support_response, conversation_id, brand.
        """
        if not conversations:
            raise ValueError("No conversations provided to build index.")

        self.metadata = conversations
        texts = [c.get("customer_message", "") for c in conversations]
        
        self.vectors = self.vectorizer.fit_transform(texts).toarray()
        self.is_built = True
        return self

    def search(self, query: str, top_k: int = 3, threshold: float = 0.55) -> List[Dict[str, Any]]:
        """
        Searches the index for historical conversations similar to the query customer message.
        Filters results by similarity_threshold.
        """
        if not self.is_built or self.vectors is None:
            return []

        query_vec = self.vectorizer.transform([query]).toarray()
        sim_scores = cosine_similarity(query_vec, self.vectors)[0]

        top_indices = sim_scores.argsort()[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(sim_scores[idx])
            if score >= threshold:
                item = self.metadata[idx].copy()
                item["similarity_score"] = round(score, 4)
                results.append(item)

        return results

    def save(self, index_path: str = "data/processed/retrieval_index.pkl"):
        """Saves vector index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "vectors": self.vectors,
                "metadata": self.metadata,
                "is_built": self.is_built
            }, f)

    def load(self, index_path: str = "data/processed/retrieval_index.pkl"):
        """Loads vector index and metadata from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Retrieval index not found at {index_path}")
        with open(index_path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.vectors = data["vectors"]
            self.metadata = data["metadata"]
            self.is_built = data["is_built"]
        return self
