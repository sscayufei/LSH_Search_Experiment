import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.benchmark as benchmark
from src.benchmark_metrics import query_metrics, summarize
from src.exact_hash import ExactHashIndex
from src.random_hyperplane_lsh import RandomHyperplaneLSH
from src.retrieval import HybridRetriever
from src.tune_parameters import deep_size


def test_query_metrics_sets_and_undefined_recall():
    assert query_metrics([1, 1, 2, 9], [1, 2, 3], 4, 10, 4) == {
        "ground_truth_count": 3, "candidate_count": 4, "result_count": 3,
        "recall": 2 / 3, "candidate_ratio": 0.4, "comparisons": 4}
    assert query_metrics([], [1], 0, 10, 0)["recall"] == 0
    assert query_metrics([], [], 0, 10, 0)["recall"] is None
    with pytest.raises(ValueError):
        query_metrics([], [], 11, 10, 0)


def test_group_aggregation_retains_empty_queries_and_uses_raw_timing_percentiles():
    rows = [{"method": "Hash + LSH", "query_id": "s0", "kind": "semantic", "recall": 0.5, "candidate_ratio": 0.1, "candidate_count": 1, "comparisons": 1},
            {"method": "Hash + LSH", "query_id": "s1", "kind": "semantic", "recall": None, "candidate_ratio": 0.2, "candidate_count": 2, "comparisons": 2},
            {"method": "Hash + LSH", "query_id": "e0", "kind": "exact", "recall": 0.25, "candidate_ratio": 0.1, "candidate_count": 1, "comparisons": 0}]
    timings = [{"method": "Hash + LSH", "query_id": q, "time_ms": ms} for q, ms in (("s0", 1), ("s0", 3), ("s1", 5), ("s1", 7), ("e0", 0.1), ("e0", 0.3))]
    resources = {"Hash + LSH": {"index_bytes": 1024, "build_seconds": 0.1}}
    summaries = summarize(rows, timings, resources)
    semantic, exact, mixed = summaries
    assert semantic["queries"] == 2 and semantic["nonempty_ground_truth_queries"] == 1
    assert semantic["recall"] == 0.5
    assert semantic["mean_candidate_ratio"] == pytest.approx(0.15)
    assert semantic["mean_time_ms"] == 4
    assert semantic["p95_time_ms"] == np.percentile([1, 3, 5, 7], 95)
    assert exact["recall"] == 0.25 and exact["mean_comparisons"] == 0
    assert mixed["recall"] == 0.375 and mixed["queries"] == 3
    rows[0]["recall"] = None
    assert summarize(rows, timings, resources)[0]["recall"] is None


def test_exact_memory_excludes_shared_text_buffers():
    short = ExactHashIndex(["a", "b"])
    long = ExactHashIndex(["a" * 10000, "b" * 10000])
    assert benchmark.exact_index_bytes(short) == benchmark.exact_index_bytes(long) > 0
    lsh = RandomHyperplaneLSH(2, 2).fit([[1, 0], [0, 1]])
    hybrid = HybridRetriever(["a", "b"], lsh)
    exact_bytes = benchmark.exact_index_bytes(hybrid._exact)
    lsh_bytes = deep_size(lsh._planes, lsh._tables)
    combined = benchmark.hybrid_index_bytes(hybrid, lsh)
    assert max(exact_bytes, lsh_bytes) <= combined <= exact_bytes + lsh_bytes


def test_count_adapters_and_hybrid_early_return():
    brute = benchmark.result_details("Brute-force", np.array([1, 2]), 10)
    assert brute["candidate_ids"] is None and brute["comparisons"] == brute["candidate_count"] == 10
    exact = benchmark.result_details("Exact Hash Only", [1], 10)
    assert exact["comparisons"] == 0 and exact["candidate_count"] == 1
    lsh = RandomHyperplaneLSH(1, 1).fit([[1.], [-1.]])
    hybrid = HybridRetriever(["original", "opposite"], lsh)
    result = benchmark.result_details("Hash + LSH", hybrid.search("original", None, 0.75), 2)
    benchmark.verify_result("Hash + LSH", result, {"kind": "exact"}, [0, 1], [0])
    assert result["comparisons"] == 0 and result["ids"] == [0]
    # Exact-first may omit other cosine-related IDs; its Recall must reflect that.
    assert query_metrics(result["ids"], [0, 1], 1, 2, 0)["recall"] == 0.5
    result["ids"] = [0, 1]
    with pytest.raises(AssertionError):
        benchmark.verify_result("Hash + LSH", result, {"kind": "exact"}, [0, 1], [0])


def test_frozen_config_checksum_and_parameter_changes_rejected(monkeypatch):
    cfg = {"seed": 42, "retrieval": {"tau": 0.75, "k": 12, "L": 32}, "selection": {"split": "validation", "phase": 4}}
    manifest = {"outputs": {str(Path("configs/frozen.yaml")): {"sha256": "fixed"}},
                "selection": {"selected": {"seed": 42, "tau": 0.75, "k": 12, "L": 32}}}

    def read_text(path, **kwargs):
        return json.dumps(manifest) if path.suffix == ".json" else yaml.safe_dump(cfg)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(benchmark, "checksum", lambda path: "fixed")
    assert benchmark.load_frozen_config("configs/frozen.yaml", "phase4.json")[0] == cfg
    cfg["retrieval"]["tau"] = 0.8
    with pytest.raises(ValueError, match="Parameters differ"):
        benchmark.load_frozen_config("configs/frozen.yaml", "phase4.json")
    monkeypatch.setattr(benchmark, "checksum", lambda path: "changed")
    with pytest.raises(ValueError, match="checksum"):
        benchmark.load_frozen_config("configs/frozen.yaml", "phase4.json")
