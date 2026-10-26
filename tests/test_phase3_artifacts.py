"""Offline checks of the actual Phase 3 index and all validation queries."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.ground_truth import BruteForceSearch, load_jsonl
from src.prepare_data import checksum
from src.random_hyperplane_lsh import RandomHyperplaneLSH, candidate_probability

OUTPUT = Path("data/lsh_validation")
pytestmark = pytest.mark.skipif(not (OUTPUT / "summary.json").exists(), reason="Real Phase 3 artifacts not generated")


def test_real_index_persistence_reproduction_and_membership():
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    for filename, digest in {**summary["inputs"], **summary["code_sha256"]}.items():
        assert checksum(filename) == digest
    for filename, info in summary["outputs"].items():
        assert checksum(OUTPUT / filename) == info["sha256"]
    config = summary["configuration"]
    x = np.load("data/processed/corpus_embeddings.npy", allow_pickle=False)
    index = RandomHyperplaneLSH(config["k"], config["L"], config["seed"]).fit(x)
    np.testing.assert_array_equal(index.hyperplanes, np.load(OUTPUT / "hyperplanes.npy", allow_pickle=False))
    saved_tables = json.loads((OUTPUT / "tables.json").read_text(encoding="utf-8"))
    assert len(saved_tables) == config["L"]
    for actual, saved in zip(index.tables, saved_tables, strict=True):
        assert {str(key): list(ids) for key, ids in actual.items()} == saved
        assert sorted(i for ids in saved.values() for i in ids) == list(range(len(x)))
    assert index.bucket_stats() == json.loads((OUTPUT / "bucket_stats.json").read_text(encoding="utf-8"))
    assert index.fit_stats["id_references"] == len(x) * config["L"]
    assert summary["theoretical_candidate_probability_at_tau"] == candidate_probability(config["tau"], config["k"], config["L"])


def test_real_queries_equal_saved_candidates_intersect_independent_ground_truth():
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    config = summary["configuration"]
    x = np.load("data/processed/corpus_embeddings.npy", allow_pickle=False)
    q = np.load("data/processed/validation_queries_embeddings.npy", allow_pickle=False)
    records = load_jsonl(OUTPUT / "queries.jsonl")
    queries = load_jsonl("data/processed/validation_queries.jsonl")
    corpus = load_jsonl("data/processed/corpus.jsonl")
    index = RandomHyperplaneLSH(config["k"], config["L"], config["seed"]).fit(x)
    oracle = BruteForceSearch(x)
    assert len(records) == len(queries) == 60
    for record, item, vector in zip(records, queries, q, strict=True):
        assert record["query_id"] == item["query_id"]
        assert record["kind"] == item["kind"]
        candidates = index.candidates(vector).tolist()
        truth = oracle.search(vector, config["tau"]).tolist()
        assert record["candidate_ids"] == candidates
        assert record["ground_truth_ids"] == truth
        assert record["lsh_ids"] == sorted(set(candidates) & set(truth))
        assert record["lsh_ids"] == index.search(vector, config["tau"]).tolist()
        np.testing.assert_array_equal(record["lsh_scores"], oracle.scores(vector)[record["lsh_ids"]])
        assert record["lsh_stats"]["candidate_count"] == record["lsh_stats"]["comparisons"] == len(candidates)
        assert record["lsh_stats"]["table_lookups"] == config["L"]
        if item["kind"] == "exact":
            assert record["hybrid_ids"] == [i for i, r in enumerate(corpus) if r["text"] == item["text"]]
            assert record["hybrid_route"] == "exact"
            assert record["hybrid_stats"]["comparisons"] == record["hybrid_stats"]["table_lookups"] == 0
        else:
            assert record["hybrid_route"] == "lsh"
            assert record["hybrid_ids"] == record["lsh_ids"]
    assert summary["total_lsh_candidates"] == summary["total_lsh_comparisons"] == sum(len(r["candidate_ids"]) for r in records)
