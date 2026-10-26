"""Real Phase 6 nested corpus, model samples, scale results and prediction audit."""
import csv
import gc
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.benchmark import hybrid_index_bytes
from src.ground_truth import BruteForceSearch, load_jsonl
from src.prepare_data import checksum
from src.prepare_scale_data import build_nested_records, verify_scale_data
from src.random_hyperplane_lsh import RandomHyperplaneLSH
from src.retrieval import HybridRetriever
from src.scale_analysis import prediction_rows

RAW = Path("results/raw")
pytestmark = pytest.mark.skipif(not (RAW / "scale_manifest.json").exists(), reason="Real Phase 6 analysis not run")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_real_scale_corpus_provenance_prefix_and_query_exclusion():
    from datasets import Dataset

    manifest, records, vectors, error = verify_scale_data("data/scale")
    assert len(records) == 100000 and error <= 1e-4
    assert manifest["code_sha256"] == checksum("src/prepare_scale_data.py")
    for filename, digest in manifest["inputs"].items():
        assert checksum(filename) == digest
    base = load_jsonl("data/processed/corpus.jsonl")
    assert records[:10000] == base
    np.testing.assert_array_equal(vectors[:10000], np.load("data/processed/corpus_embeddings.npy"))
    excluded = {q["text"] for split in ("validation", "test") for q in load_jsonl(f"data/processed/{split}_queries.jsonl")}
    assert not {r["text"] for r in records[10000:]} & excluded
    arrow = next(Path("data/raw/datasets").rglob("41f699770310302022a4dd75d4cf903bfef9ea46/*.arrow"))
    columns = Dataset.from_file(str(arrow)).to_dict()
    rows = ({"anchor": a, "positive": p} for a, p in zip(columns["anchor"], columns["positive"], strict=True))
    rebuilt, available = build_nested_records(base, rows, excluded, 100000, manifest["seed"])
    assert available == manifest["available_unique_texts"] == 149446
    assert rebuilt == records
    for record in records[10000:]:
        assert columns[record["source_column"]][record["source_pair_id"]] == record["text"]


def test_saved_extension_vectors_match_offline_real_model_samples():
    import torch
    from sentence_transformers import SentenceTransformer

    config = yaml.safe_load(Path("configs/selected_experiment.yaml").read_text(encoding="utf-8"))
    records = load_jsonl("data/scale/corpus.jsonl")
    vectors = np.load("data/scale/corpus_embeddings.npy", mmap_mode="r", allow_pickle=False)
    torch.set_num_threads(4)
    model = SentenceTransformer(config["embedding"]["model"], revision=config["embedding"]["revision"], device="cpu", local_files_only=True,
                                cache_folder=str(Path(config["cache"]) / "models"))
    indices = [10000, 49999, 99999]
    actual = model.encode([records[i]["text"] for i in indices], normalize_embeddings=True, convert_to_numpy=True)
    np.testing.assert_allclose(vectors[indices], actual, atol=2e-6, rtol=1e-4)


def test_all_scale_metrics_ground_truth_nesting_and_baseline_match_phase5():
    config = yaml.safe_load(Path("configs/selected_experiment.yaml").read_text(encoding="utf-8"))
    plan = yaml.safe_load(Path("configs/scale.yaml").read_text(encoding="utf-8"))
    data = load_jsonl("data/scale/corpus.jsonl")
    vectors = np.load("data/scale/corpus_embeddings.npy", mmap_mode="r", allow_pickle=False)
    all_queries = load_jsonl("data/processed/test_queries.jsonl")
    query_indices = [i for i, r in enumerate(all_queries) if r["kind"] == "semantic"]
    queries = [all_queries[i] for i in query_indices]
    Q = np.load("data/processed/test_queries_embeddings.npy")[query_indices]
    summaries = read_csv(RAW / "scale_summary.csv")
    assert [int(r["N"]) for r in summaries] == [10000, 50000, 100000]
    previous = None
    for summary in summaries:
        N = int(summary["N"])
        directory = RAW / f"scale_n{N}"
        records = load_jsonl(directory / "results.jsonl")
        per_query = read_csv(directory / "per_query.csv")
        timings = read_csv(directory / "timings.csv")
        assert len(records) == len(per_query) == 100 and len(timings) == 100 * plan["repeats"]
        # Release the full-scan working matrix before fitting the index, as in workers.
        oracle = BruteForceSearch(vectors[:N])
        truths = [oracle.search(q, config["retrieval"]["tau"]).tolist() for q in Q]
        del oracle
        gc.collect()
        index = RandomHyperplaneLSH(config["retrieval"]["k"], config["retrieval"]["L"], config["seed"]).fit(vectors[:N])
        hybrid = HybridRetriever((r["text"] for r in data[:N]), index)
        recalls, candidate_counts, false_counts, times = [], [], [], []
        for i, (query, saved, row, truth) in enumerate(zip(queries, records, per_query, truths, strict=True)):
            assert query["query_id"] == saved["query_id"] == row["query_id"]
            actual = hybrid.search(query["text"], Q[i], config["retrieval"]["tau"])
            assert actual.route == "lsh"
            assert saved["ground_truth_ids"] == truth
            assert saved["candidate_ids"] == actual.candidate_ids.tolist()
            assert saved["result_ids"] == actual.ids.tolist() == sorted(set(saved["candidate_ids"]) & set(truth))
            c, f = len(saved["candidate_ids"]), len(set(saved["candidate_ids"]) - set(truth))
            assert int(row["candidate_count"]) == int(row["comparisons"]) == c
            assert int(row["false_candidate_count"]) == f
            candidate_counts.append(c)
            false_counts.append(f)
            if truth:
                value = len(saved["result_ids"]) / len(truth)
                assert float(row["recall"]) == value
                recalls.append(value)
            else:
                assert row["recall"] == ""
            samples = [float(t["time_ms"]) for t in timings if t["query_id"] == query["query_id"]]
            assert len(samples) == plan["repeats"]
            assert float(row["mean_time_ms"]) == np.mean(samples)
            assert float(row["p95_time_ms"]) == np.percentile(samples, 95)
        assert float(summary["recall"]) == np.mean(recalls)
        assert int(summary["nonempty_ground_truth_queries"]) == len(recalls)
        assert float(summary["mean_candidates"]) == float(summary["mean_comparisons"]) == np.mean(candidate_counts)
        assert float(summary["mean_false_candidates"]) == np.mean(false_counts)
        assert float(summary["mean_candidate_ratio"]) == np.mean([c / N for c in candidate_counts])
        assert float(summary["mean_time_ms"]) == np.mean([float(t["time_ms"]) for t in timings])
        assert float(summary["p95_time_ms"]) == np.percentile([float(t["time_ms"]) for t in timings], 95)
        assert int(summary["index_bytes"]) == hybrid_index_bytes(hybrid, index)
        buckets = json.loads((directory / "bucket_stats.json").read_text(encoding="utf-8"))
        assert len(buckets) == config["retrieval"]["L"] and all(row["id_references"] == N for row in buckets)
        if previous is not None:
            for small, large in zip(previous, records, strict=True):
                assert set(small["candidate_ids"]) <= set(large["candidate_ids"])
                assert set(small["ground_truth_ids"]) <= set(large["ground_truth_ids"])
        else:
            phase5 = {r["query_id"]: r for r in load_jsonl(RAW / "test_results.jsonl") if r["method"] == "Hash + LSH" and r["kind"] == "semantic"}
            for row in records:
                assert row["candidate_ids"] == phase5[row["query_id"]]["candidate_ids"]
                assert row["result_ids"] == phase5[row["query_id"]]["ids"]
        previous = records
        del hybrid, index
        gc.collect()
    assert len({r["planes_sha256"] for r in summaries}) == 1


def test_scale_checksums_and_prediction_classification():
    manifest = json.loads((RAW / "scale_manifest.json").read_text(encoding="utf-8"))
    for name, digest in {**manifest["inputs"], **manifest["code_sha256"]}.items():
        assert checksum(name) == digest
    for name, info in manifest["outputs"].items():
        assert checksum(name) == info["sha256"]
    assert manifest["million_size_measured"] is False
    assert 1000000 not in manifest["measured_sizes"]
    plan = yaml.safe_load(Path("configs/scale.yaml").read_text(encoding="utf-8"))
    baseline = json.loads((RAW / "scale_n10000/summary.json").read_text(encoding="utf-8"))
    expected = prediction_rows(baseline, plan["prediction_sizes"], plan["reference_similarities"], 12, 32, 384)
    actual = read_csv(RAW / "scale_predictions.csv")
    assert len(actual) == len(expected)
    for saved, rebuilt in zip(actual, expected, strict=True):
        assert saved["value_type"] == rebuilt["value_type"] != "measured"
        assert float(saved["value"]) == rebuilt["value"]
        assert saved["units"] == rebuilt["units"]
