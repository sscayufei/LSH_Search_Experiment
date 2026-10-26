"""Audit final real test results, metric aggregation and frozen configuration."""
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.benchmark import exact_index_bytes, hybrid_index_bytes
from src.benchmark_metrics import METHODS
from src.exact_hash import ExactHashIndex
from src.ground_truth import load_jsonl
from src.prepare_data import checksum
from src.random_hyperplane_lsh import RandomHyperplaneLSH
from src.retrieval import HybridRetriever

RAW = Path("results/raw")
pytestmark = pytest.mark.skipif(not (RAW / "phase5_manifest.json").exists(), reason="Real Phase 5 benchmark not run")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_phase5_provenance_and_config_frozen_before_and_after_test():
    manifest = json.loads((RAW / "phase5_manifest.json").read_text(encoding="utf-8"))
    for filename, digest in {**manifest["inputs"], **manifest["code_sha256"]}.items():
        assert checksum(filename) == digest
    for filename, info in manifest["outputs"].items():
        assert checksum(filename) == info["sha256"]
    config = yaml.safe_load(Path("configs/selected_experiment.yaml").read_text(encoding="utf-8"))
    env = json.loads((RAW / "environment.json").read_text(encoding="utf-8"))
    plan = json.loads((RAW / "benchmark_plan.json").read_text(encoding="utf-8"))
    assert env["config_sha256_before"] == env["config_sha256_after"] == checksum("configs/selected_experiment.yaml")
    assert env["frozen_config"] == config
    assert env["test_evaluation_run"] is True
    assert plan["frozen_config_sha256"] == manifest["frozen_config_sha256"]
    assert datetime.fromisoformat(config["selection"]["frozen_utc"]) < datetime.fromisoformat(plan["started_utc"])
    assert manifest["corpus_size"] == 10000 and manifest["query_count"] == 120
    texts = [r["text"] for r in load_jsonl("data/processed/corpus.jsonl")]
    x = np.load("data/processed/corpus_embeddings.npy", allow_pickle=False)
    lsh = RandomHyperplaneLSH(config["retrieval"]["k"], config["retrieval"]["L"], config["seed"]).fit(x)
    assert env["resources"]["Brute-force"]["index_bytes"] == 0
    assert env["resources"]["Exact Hash Only"]["index_bytes"] == exact_index_bytes(ExactHashIndex(texts))
    assert env["resources"]["Hash + LSH"]["index_bytes"] == hybrid_index_bytes(HybridRetriever(texts, lsh), lsh)
    assert env["shared_storage"]["normalized_corpus_workspace_bytes_by_method"]["Brute-force"] == x.shape[0] * x.shape[1] * 8


def test_all_test_ground_truth_and_results_against_independent_cosine_formula():
    config = yaml.safe_load(Path("configs/selected_experiment.yaml").read_text(encoding="utf-8"))
    tau = config["retrieval"]["tau"]
    corpus = load_jsonl("data/processed/corpus.jsonl")
    queries = load_jsonl("data/processed/test_queries.jsonl")
    X = np.load("data/processed/corpus_embeddings.npy").astype(np.float64)
    Q = np.load("data/processed/test_queries_embeddings.npy").astype(np.float64)
    expected_scores = (Q @ X.T) / (np.linalg.norm(Q, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None, :])
    scores = np.load(RAW / "test_scores.npy", allow_pickle=False)
    assert scores.shape == (120, 10000) and scores.dtype == np.float64
    np.testing.assert_allclose(scores, np.clip(expected_scores, -1, 1), atol=2e-14, rtol=1e-12)
    ground_truth = load_jsonl(RAW / "test_ground_truth.jsonl")
    results = load_jsonl(RAW / "test_results.jsonl")
    metadata = read_csv(RAW / "test_queries.csv")
    per_query = read_csv(RAW / "test_per_query.csv")
    assert len(results) == len(per_query) == 360 and len(ground_truth) == len(metadata) == 120
    assert sum(r["kind"] == "semantic" for r in queries) == 100
    index = RandomHyperplaneLSH(config["retrieval"]["k"], config["retrieval"]["L"], config["seed"]).fit(X)
    for i, (query, gt, info) in enumerate(zip(queries, ground_truth, metadata, strict=True)):
        truth = np.flatnonzero(expected_scores[i] >= tau).tolist()
        assert query["query_id"] == gt["query_id"] == info["query_id"]
        assert gt["ground_truth_ids"] == truth
        np.testing.assert_array_equal(gt["ground_truth_scores"], scores[i, truth])
        exact = [j for j, row in enumerate(corpus) if row["text"] == query["text"]]
        assert json.loads(info["exact_ids"]) == exact
        assert (info["exact_hit"] == "True") == bool(exact)
        for method in METHODS:
            saved = next(r for r in results if r["method"] == method and r["query_id"] == query["query_id"])
            measured = next(r for r in per_query if r["method"] == method and r["query_id"] == query["query_id"])
            if method == "Brute-force":
                expected_ids, count, comparisons = truth, len(X), len(X)
                assert saved["candidate_scope"] == "all_corpus" and saved["candidate_ids"] is None
            elif method == "Exact Hash Only" or exact:
                expected_ids, count, comparisons = exact, len(exact), 0
                assert saved["candidate_ids"] == exact
            else:
                candidates = index.candidates(Q[i]).tolist()
                expected_ids, count, comparisons = sorted(set(candidates) & set(truth)), len(candidates), len(candidates)
                assert saved["candidate_ids"] == candidates
            assert saved["ids"] == expected_ids
            assert int(measured["candidate_count"]) == count
            assert int(measured["comparisons"]) == comparisons
            assert int(measured["result_count"]) == len(expected_ids)
            assert int(measured["ground_truth_count"]) == len(truth)
            if truth:
                assert float(measured["recall"]) == len(set(expected_ids) & set(truth)) / len(truth)
            else:
                assert measured["recall"] == ""
    assert sum(r["kind"] == "semantic" and not r["ground_truth_ids"] for r in ground_truth) == 5


def test_every_summary_metric_traceable_to_query_rows_and_all_repeated_samples():
    per_query = read_csv(RAW / "test_per_query.csv")
    timings = read_csv(RAW / "test_timings.csv")
    summary = read_csv(RAW / "method_summary.csv")
    config = yaml.safe_load(Path("configs/selected_experiment.yaml").read_text(encoding="utf-8"))
    env = json.loads((RAW / "environment.json").read_text(encoding="utf-8"))
    queries = load_jsonl("data/processed/test_queries.jsonl")
    repeats = env["measurement"]["repeats"]
    assert len(timings) == 120 * 3 * repeats == 1800 and len(summary) == 9
    for row in per_query:
        samples = [r for r in timings if r["method"] == row["method"] and r["query_id"] == row["query_id"]]
        assert len(samples) == repeats
        assert {int(r["repeat"]) for r in samples} == set(range(repeats))
        assert all(r["candidate_count"] == row["candidate_count"] and r["comparisons"] == row["comparisons"] for r in samples)
        values = [float(r["time_ms"]) for r in samples]
        assert float(row["time_ms"]) == np.mean(values)
        assert float(row["p50_time_ms"]) == np.percentile(values, 50)
        assert float(row["p95_time_ms"]) == np.percentile(values, 95)
    for row in summary:
        group = [r for r in per_query if r["method"] == row["method"] and (row["group"] == "all" or r["kind"] == row["group"])]
        ids = {r["query_id"] for r in group}
        samples = [float(r["time_ms"]) for r in timings if r["method"] == row["method"] and r["query_id"] in ids]
        recalls = [float(r["recall"]) for r in group if r["recall"] != ""]
        assert int(row["queries"]) == len(group)
        assert int(row["nonempty_ground_truth_queries"]) == len(recalls)
        assert int(row["empty_ground_truth_queries"]) == len(group) - len(recalls)
        assert float(row["recall"]) == np.mean(recalls)
        assert float(row["mean_candidate_ratio"]) == np.mean([float(r["candidate_ratio"]) for r in group])
        assert float(row["p95_candidate_ratio"]) == np.percentile([float(r["candidate_ratio"]) for r in group], 95)
        assert float(row["mean_comparisons"]) == np.mean([int(r["comparisons"]) for r in group])
        assert float(row["mean_time_ms"]) == np.mean(samples)
        assert float(row["p50_time_ms"]) == np.percentile(samples, 50)
        assert float(row["p95_time_ms"]) == np.percentile(samples, 95)
        assert int(row["index_bytes"]) == env["resources"][row["method"]]["index_bytes"]
    # Independently regenerate only the seeded call schedule, not any measured times.
    rng = np.random.default_rng(config["seed"])
    expected_schedule = []
    for repeat in range(-env["measurement"]["warmup_rounds"], repeats):
        for i in rng.permutation(len(queries)):
            for method in rng.permutation(METHODS):
                if repeat >= 0:
                    expected_schedule.append((repeat, str(method), queries[i]["query_id"]))
    assert [(int(r["repeat"]), r["method"], r["query_id"]) for r in timings] == expected_schedule
