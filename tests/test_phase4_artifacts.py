"""Audit the real validation grid and frozen settings without opening test data."""
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ground_truth import BruteForceSearch
from src.metrics import recall
from src.prepare_data import checksum
from src.random_hyperplane_lsh import RandomHyperplaneLSH
from src.tune_parameters import analyze_thresholds, deep_size, load_validation, select_configuration

RAW = Path("results/raw")
pytestmark = pytest.mark.skipif(not (RAW / "phase4_manifest.json").exists(), reason="Real Phase 4 artifacts not generated")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_phase4_provenance_threshold_analysis_and_lock():
    manifest = json.loads((RAW / "phase4_manifest.json").read_text(encoding="utf-8"))
    for filename, digest in {**manifest["inputs"], **manifest["code_sha256"]}.items():
        assert checksum(filename) == digest
    for filename, info in manifest["outputs"].items():
        assert checksum(filename) == info["sha256"]
    assert not any("test_queries" in name for name in manifest["inputs"])
    plan = yaml.safe_load(Path("configs/phase4.yaml").read_text(encoding="utf-8"))
    frozen = yaml.safe_load(Path("configs/selected_experiment.yaml").read_text(encoding="utf-8"))
    snapshot = json.loads((RAW / "phase4_plan.json").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(snapshot["created_utc"]) < datetime.fromisoformat(frozen["selection"]["frozen_utc"])
    assert frozen["selection"]["test_evaluation_run"] is False
    assert frozen["selection"]["grid_sha256"] == checksum(RAW / "validation_grid.csv")
    assert frozen["selection"]["threshold_analysis_sha256"] == checksum(RAW / "threshold_analysis.json")
    assert frozen["selection"]["policy_sha256"] == checksum("configs/phase4.yaml")
    _, queries, X, Q, _ = load_validation("data/processed")
    oracle = BruteForceSearch(X)
    scores = np.stack([oracle.scores(q) for q in Q])
    np.testing.assert_array_equal(scores, np.load(RAW / "validation_semantic_scores.npy", allow_pickle=False))
    analysis, refs = analyze_thresholds(queries, scores, plan["thresholds"], plan["threshold_policy"], frozen["seed"])
    assert analysis == json.loads((RAW / "threshold_analysis.json").read_text(encoding="utf-8"))
    saved_refs = read_csv(RAW / "threshold_reference_pairs.csv")
    assert len(saved_refs) == len(refs)
    for saved, rebuilt in zip(saved_refs, refs, strict=True):
        assert saved["query_id"] == rebuilt["query_id"] and saved["kind"] == rebuilt["kind"]
        assert int(saved["corpus_id"]) == rebuilt["corpus_id"]
        assert float(saved["score"]) == rebuilt["score"]
    assert frozen["retrieval"]["tau"] == analysis["selected_tau"]
    selected = manifest["selection"]["selected"]
    assert frozen["retrieval"] == {key: selected[key] for key in ("tau", "k", "L")}
    assert frozen["seed"] == selected["seed"]


def test_every_grid_row_matches_rebuilt_index_and_raw_queries():
    plan = yaml.safe_load(Path("configs/phase4.yaml").read_text(encoding="utf-8"))
    _, queries, X, Q, _ = load_validation("data/processed")
    grid = read_csv(RAW / "validation_grid.csv")
    per_query = read_csv(RAW / "validation_per_query.csv")
    timings = read_csv(RAW / "validation_timings.csv")
    assert {(int(r["k"]), int(r["L"])) for r in grid} == {(k, L) for k in plan["grid"]["k"] for L in plan["grid"]["L"]}
    assert len(grid) == 12 and len(per_query) == 12 * 50 and len(timings) == 12 * 50 * plan["measurement"]["repeats"]
    oracle = BruteForceSearch(X)
    score_matrix = np.stack([oracle.scores(q) for q in Q])
    typed_grid = []
    for row in grid:
        k, L, tau, seed = int(row["k"]), int(row["L"]), float(row["tau"]), int(row["seed"])
        group = [r for r in per_query if int(r["k"]) == k and int(r["L"]) == L]
        index = RandomHyperplaneLSH(k, L, seed).fit(X)
        recalls, ratios, means = [], [], []
        for i, (record, query, q) in enumerate(zip(group, queries, Q, strict=True)):
            result = index.search_with_stats(q, tau)
            truth = np.flatnonzero(score_matrix[i] >= tau)
            assert record["query_id"] == query["query_id"]
            assert int(record["ground_truth_count"]) == len(truth)
            assert int(record["candidate_count"]) == int(record["comparisons"]) == len(result.candidate_ids)
            assert int(record["result_count"]) == len(result.ids)
            assert result.ids.tolist() == sorted(set(result.candidate_ids) & set(truth))
            value = recall(result.ids, truth)
            assert (record["recall"] == "") if value is None else float(record["recall"]) == value
            if value is not None:
                recalls.append(value)
            ratios.append(len(result.candidate_ids) / len(X))
            samples = [r for r in timings if int(r["k"]) == k and int(r["L"]) == L and r["query_id"] == query["query_id"]]
            assert len(samples) == plan["measurement"]["repeats"]
            assert all(int(r["comparisons"]) == len(result.candidate_ids) for r in samples)
            mean_time = float(np.mean([float(r["time_ms"]) for r in samples]))
            assert float(record["mean_time_ms"]) == mean_time
            means.append(mean_time)
        assert float(row["recall"]) == np.mean(recalls)
        assert int(row["nonempty_ground_truth_queries"]) == len(recalls)
        assert int(row["empty_ground_truth_queries"]) == len(queries) - len(recalls)
        assert float(row["mean_candidate_ratio"]) == np.mean(ratios)
        assert float(row["p95_candidate_ratio"]) == np.percentile(ratios, 95)
        assert float(row["mean_time_ms"]) == np.mean(means)
        assert float(row["p95_time_ms"]) == np.percentile(means, 95)
        assert int(row["index_bytes"]) == deep_size(index._planes, index._tables)
        typed_grid.append({key: int(row[key]) if key in ("k", "L", "index_bytes") else float(row[key])
                           for key in ("k", "L", "index_bytes", "recall", "mean_candidate_ratio", "p95_candidate_ratio")})
    selected, feasible = select_configuration(typed_grid, plan["selection"])
    actual = next(r for r in grid if r["selected"] == "True")
    assert (selected["k"], selected["L"]) == (int(actual["k"]), int(actual["L"]))
    assert feasible == json.loads((RAW / "parameter_selection.json").read_text(encoding="utf-8"))["budget_satisfied"]
