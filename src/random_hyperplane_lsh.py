"""Random hyperplane LSH: independent k-bit tables and candidate-only cosine."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from time import perf_counter
from types import MappingProxyType

import numpy as np

from src.ground_truth import cosine_scores_normalized, normalize_rows, threshold_ids


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def candidate_probability(similarity, k, L):
    """Theoretical independent-table probability, not measured Recall."""
    k, L = positive_integer(k, "k"), positive_integer(L, "L")
    if not isinstance(similarity, Real) or not np.isfinite(similarity) or not -1 <= similarity <= 1:
        raise ValueError("Similarity must be finite and in [-1,1]")
    p = 1 - math.acos(similarity) / math.pi
    single_table = p ** k
    if single_table == 1:
        return 1.0
    return -math.expm1(L * math.log1p(-single_table))


def pack_signatures(bits):
    """Bit j has weight 2**j; Python integers also support k > 64."""
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.dtype != np.bool_:
        raise ValueError("Expected a two-dimensional boolean bit matrix")
    packed = np.packbits(bits, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


@dataclass(frozen=True)
class SearchResult:
    ids: np.ndarray
    candidate_ids: np.ndarray
    scores: np.ndarray | None
    route: str
    stats: dict


class RandomHyperplaneLSH:
    def __init__(self, k=12, L=16, random_state=42):
        self._k = positive_integer(k, "k")
        self._L = positive_integer(L, "L")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        self._seed = int(random_state)
        self._vectors = None

    @property
    def k(self):
        return self._k

    @property
    def L(self):
        return self._L

    def _require_fit(self):
        if self._vectors is None:
            raise RuntimeError("Call fit(X) before querying the index")

    @property
    def n_samples(self):
        self._require_fit()
        return len(self._vectors)

    @property
    def hyperplanes(self):
        self._require_fit()
        return self._planes.copy()

    @property
    def tables(self):
        self._require_fit()
        return tuple(MappingProxyType(table) for table in self._tables)

    @property
    def fit_stats(self):
        self._require_fit()
        return dict(self._fit_stats)

    def fit(self, X):
        start = perf_counter()
        vectors = normalize_rows(X)
        normalization_seconds = perf_counter() - start
        planes = np.random.default_rng(self._seed).normal(size=(self.L, self.k, vectors.shape[1]))
        tables = []
        signature_seconds = bucket_seconds = 0.0
        for table_planes in planes:
            tick = perf_counter()
            bits = np.einsum("nd,kd->nk", vectors, table_planes, optimize=False) >= 0
            signatures = pack_signatures(bits)
            signature_seconds += perf_counter() - tick
            tick = perf_counter()
            table = {}
            for doc_id, signature in enumerate(signatures):
                table.setdefault(signature, []).append(doc_id)
            tables.append({key: tuple(ids) for key, ids in table.items()})
            bucket_seconds += perf_counter() - tick
        vectors.flags.writeable = False
        planes.flags.writeable = False
        self._vectors, self._planes, self._tables = vectors, planes, tables
        self._fit_stats = {"build_seconds": perf_counter() - start, "normalization_seconds": normalization_seconds,
                           "signature_seconds": signature_seconds, "bucket_build_seconds": bucket_seconds,
                           "id_references": len(vectors) * self.L}
        return self

    def _query_vector(self, query):
        self._require_fit()
        query = np.asarray(query)
        if query.ndim != 1 or query.shape[0] != self._vectors.shape[1]:
            raise ValueError("Query must be a vector matching corpus dimension")
        return normalize_rows(query[None, :])[0]

    def _signatures(self, normalized_query):
        return tuple(pack_signatures(np.einsum("nd,kd->nk", normalized_query[None, :], table, optimize=False) >= 0)[0]
                     for table in self._planes)

    def signatures(self, query):
        return self._signatures(self._query_vector(query))

    def _candidates(self, query):
        tick = perf_counter()
        normalized = self._query_vector(query)
        normalization_seconds = perf_counter() - tick
        tick = perf_counter()
        signatures = self._signatures(normalized)
        signature_seconds = perf_counter() - tick
        tick = perf_counter()
        ids = set()
        bucket_references = 0
        for table, signature in zip(self._tables, signatures, strict=True):
            bucket = table.get(signature, ())
            bucket_references += len(bucket)
            ids.update(bucket)
        candidates = np.asarray(sorted(ids), dtype=np.int64)
        stats = {"normalization_seconds": normalization_seconds, "signature_seconds": signature_seconds,
                 "bucket_lookup_seconds": perf_counter() - tick, "table_lookups": self.L,
                 "bucket_references": bucket_references, "candidate_count": len(candidates)}
        return normalized, candidates, stats

    def candidates(self, query):
        return self._candidates(query)[1]

    def _score_candidates(self, normalized_query, candidate_ids):
        # Gather only C(q). Never score self._vectors in full or call the oracle.
        return cosine_scores_normalized(self._vectors[candidate_ids], normalized_query)

    def search_with_stats(self, query, tau):
        threshold_ids(np.empty(0), tau)
        normalized, candidates, stats = self._candidates(query)
        tick = perf_counter()
        scores = self._score_candidates(normalized, candidates) if len(candidates) else np.empty(0, dtype=np.float64)
        keep = threshold_ids(scores, tau)
        stats.update(verification_seconds=perf_counter() - tick, comparisons=len(candidates))
        return SearchResult(candidates[keep], candidates, scores[keep], "lsh", stats)

    def search(self, query, tau):
        return self.search_with_stats(query, tau).ids

    def bucket_stats(self):
        self._require_fit()
        result = []
        for table_id, table in enumerate(self._tables):
            sizes = [len(ids) for ids in table.values()]
            result.append({"table": table_id, "occupied_buckets": len(table), "possible_buckets": 1 << self.k,
                           "empty_bucket_ratio": 1 - len(table) / (1 << self.k),
                           "id_references": sum(sizes), "max_bucket_size": max(sizes, default=0),
                           "mean_occupied_bucket_size": float(np.mean(sizes)) if sizes else 0.0})
        return result
