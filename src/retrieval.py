"""Exact-first retrieval; semantic misses use only LSH candidates."""
import numpy as np

from src.exact_hash import ExactHashIndex
from src.ground_truth import threshold_ids
from src.random_hyperplane_lsh import SearchResult


class HybridRetriever:
    def __init__(self, texts, lsh):
        texts = tuple(texts)
        if len(texts) != lsh.n_samples:
            raise ValueError("Texts must align with the fitted LSH corpus rows")
        self._exact = ExactHashIndex(texts)
        self._lsh = lsh

    def search(self, text, query_vector, tau):
        threshold_ids(np.empty(0), tau)
        ids = self._exact.search(text)
        if ids:
            exact_ids = np.asarray(ids, dtype=np.int64)
            return SearchResult(exact_ids, exact_ids.copy(), None, "exact",
                                {"normalization_seconds": 0.0, "signature_seconds": 0.0,
                                 "bucket_lookup_seconds": 0.0, "verification_seconds": 0.0,
                                 "table_lookups": 0, "bucket_references": 0,
                                 "candidate_count": len(ids), "comparisons": 0})
        return self._lsh.search_with_stats(query_vector, tau)
