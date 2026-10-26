import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.ground_truth import BruteForceSearch
from src.metrics import recall
import src.tune_parameters as tuning


def test_recall_sets_and_empty_ground_truth():
    assert recall([1, 1, 2, 99], [1, 2, 3]) == pytest.approx(2 / 3)
    assert recall([], [1]) == 0
    assert recall([1], []) is None
    assert recall([], []) is None


def test_threshold_rule_prefers_highest_feasible_and_has_explicit_fallback():
    rows = [{"tau": 0.7, "source_query_coverage": 0.98, "random_reference_exceedance": 0.005},
            {"tau": 0.75, "source_query_coverage": 0.92, "random_reference_exceedance": 0.002},
            {"tau": 0.8, "source_query_coverage": 0.88, "random_reference_exceedance": 0.0}]
    policy = {"min_source_query_coverage": 0.9, "max_random_reference_exceedance": 0.01}
    selected, feasible = tuning.choose_threshold(rows, policy)
    assert selected["tau"] == 0.75 and feasible
    policy["min_source_query_coverage"] = 1
    selected, feasible = tuning.choose_threshold(rows, policy)
    assert selected["tau"] == 0.7 and not feasible


def test_threshold_reference_sampling_is_reproducible_and_excludes_all_sources():
    scores = np.array([[0.9, 0.81, 0.2, 0.1], [0.1, 0.76, 0.2, 0.3], [0.1, 0.2, 0.65, 0.3]])
    queries = [{"query_id": "a", "sources": [{"corpus_id": 0}, {"corpus_id": 1}, {"corpus_id": 0}]},
               {"query_id": "b", "sources": [{"corpus_id": 1}]}, {"query_id": "c", "sources": [{"corpus_id": 2}]}]
    policy = {"min_source_query_coverage": 0.9, "max_random_reference_exceedance": 0.01, "random_references_per_query": 2}
    first, refs = tuning.analyze_thresholds(queries, scores, [0.7, 0.75, 0.8], policy, 42)
    second, repeated = tuning.analyze_thresholds(queries, scores, [0.7, 0.75, 0.8], policy, 42)
    assert first == second and refs == repeated
    assert first["source_pair_count"] == 4
    assert first["random_reference_count"] == 6
    assert first["threshold_rows"][0]["empty_ground_truth_queries"] == 1
    for ref in refs:
        row = next(i for i, query in enumerate(queries) if query["query_id"] == ref["query_id"])
        assert ref["score"] == scores[row, ref["corpus_id"]]
        if ref["kind"] == "random_non_source":
            assert ref["corpus_id"] not in {s["corpus_id"] for s in queries[row]["sources"]}


def test_validation_loader_never_opens_test_files(monkeypatch):
    calls = []
    manifest = {"files": {name: {"sha256": "ok"} for name in ("corpus.jsonl", "corpus_embeddings.npy", "validation_queries.jsonl", "validation_queries_embeddings.npy")},
                "config": {"embedding": {"dimension": 2}}}

    def guard(path):
        name = Path(path).name
        assert "test" not in name
        calls.append(name)
        return name

    def read_text(path, **kwargs):
        assert guard(path) == "manifest.json"
        return json.dumps(manifest)

    def load_records(path):
        if guard(path) == "corpus.jsonl":
            return [{"doc_id": 0, "text": "a"}, {"doc_id": 1, "text": "b"}]
        return [{"query_id": "v0", "kind": "semantic", "text": "similar a"}, {"query_id": "v1", "kind": "exact", "text": "a"}]

    def load_vectors(path, **kwargs):
        guard(path)
        return np.eye(2, dtype=np.float32)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(tuning, "checksum", lambda path: (guard(path), "ok")[1])
    monkeypatch.setattr(tuning, "load_jsonl", load_records)
    monkeypatch.setattr(tuning.np, "load", load_vectors)
    corpus, queries, X, Q, _ = tuning.load_validation("data/processed")
    assert len(corpus) == 2 and len(queries) == 1
    assert Q.shape == (1, 2)
    assert set(calls) == {"manifest.json", "corpus.jsonl", "corpus_embeddings.npy", "validation_queries.jsonl", "validation_queries_embeddings.npy"}


def test_deep_size_deduplicates_objects_and_array_buffers():
    array = np.ones((10, 3))
    assert tuning.deep_size(array, array) == sys.getsizeof(array)
    cycle = []
    cycle.append(cycle)
    assert tuning.deep_size(cycle) == sys.getsizeof(cycle)
    shared = [array, array]
    assert tuning.deep_size(shared) == sys.getsizeof(shared) + sys.getsizeof(array)


def test_probability_target_minimum_tables():
    assert tuning.min_tables_for_target(0, 1, 0.75) == 2
    assert tuning.min_tables_for_target(1, 4, 0.9) == 1
    assert tuning.min_tables_for_target(-1, 4, 0.9) is None
    L = tuning.min_tables_for_target(0.75, 12, 0.9)
    assert tuning.candidate_probability(0.75, 12, L) >= 0.9
    assert tuning.candidate_probability(0.75, 12, L - 1) < 0.9
    with pytest.raises(ValueError):
        tuning.min_tables_for_target(0, 1, 1)


def test_budget_selection_pareto_and_ties():
    rows = [{"k": 8, "L": 32, "recall": 1., "mean_candidate_ratio": 0.2, "p95_candidate_ratio": 0.3, "index_bytes": 200},
            {"k": 8, "L": 8, "recall": 0.9, "mean_candidate_ratio": 0.04, "p95_candidate_ratio": 0.08, "index_bytes": 100},
            {"k": 12, "L": 8, "recall": 0.9, "mean_candidate_ratio": 0.03, "p95_candidate_ratio": 0.07, "index_bytes": 100},
            {"k": 16, "L": 4, "recall": 0.4, "mean_candidate_ratio": 0.03, "p95_candidate_ratio": 0.06, "index_bytes": 50}]
    policy = {"max_mean_candidate_ratio": 0.05, "max_p95_candidate_ratio": 0.10, "max_index_bytes": 150}
    selected, feasible = tuning.select_configuration(rows, policy)
    assert selected is rows[2] and feasible
    assert tuning.pareto_front(rows) == [0, 2]
    policy["max_mean_candidate_ratio"] = 0.001
    selected, feasible = tuning.select_configuration(rows, policy)
    assert not feasible and selected is rows[2]


def test_measured_empty_ground_truth_retained_and_excluded_only_from_recall_mean():
    X = np.array([[1., 0], [0, 1], [-1, 0]])
    Q = np.array([[1., 0], [0, 1], [1, 1]])
    queries = [{"query_id": str(i)} for i in range(3)]
    oracle = BruteForceSearch(X)
    truths = [oracle.search(q, 1).tolist() for q in Q]
    row, per_query, timings = tuning.evaluate_configuration(X, Q, queries, truths, 2, 2, 1, 42, {"warmup_rounds": 1, "repeats": 2})
    assert row["nonempty_ground_truth_queries"] == 2
    assert row["empty_ground_truth_queries"] == 1
    assert row["recall"] == 1
    assert per_query[2]["recall"] is None and len(per_query) == 3
    assert len(timings) == 6
    assert row["mean_comparisons"] == row["mean_candidates"]
