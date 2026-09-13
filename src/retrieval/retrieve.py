"""
Retrieval wrapper module.
Exposes simple function for querying historical support resolutions.
"""

from typing import List, Dict, Any, Optional
from src.retrieval.index import ResolutionVectorStore

_GLOBAL_INDEX: Optional[ResolutionVectorStore] = None

def get_or_load_index(index_path: str = "data/processed/retrieval_index.pkl") -> ResolutionVectorStore:
    """Loads or initializes global resolution retrieval store."""
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = ResolutionVectorStore()
        try:
            _GLOBAL_INDEX.load(index_path)
        except Exception:
            pass
    return _GLOBAL_INDEX


def retrieve_similar_resolutions(
    query: str,
    top_k: int = 3,
    threshold: float = 0.55,
    index_path: str = "data/processed/retrieval_index.pkl"
) -> List[Dict[str, Any]]:
    """
    Retrieves top historical resolution matches for a customer message.
    """
    store = get_or_load_index(index_path)
    if not store.is_built:
        return []
    return store.search(query, top_k=top_k, threshold=threshold)
