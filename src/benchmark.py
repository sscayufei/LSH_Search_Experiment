"""Phase 5 frozen-parameter, precomputed-embedding test benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from src.benchmark_metrics import METHODS, query_metrics, summarize
from src.exact_hash import ExactHashIndex
from src.ground_truth import BruteForceSearch, load_jsonl
from src.plot_benchmark import plot_results
from src.prepare_data import checksum, save_json, verify_saved
from src.random_hyperplane_lsh import RandomHyperplaneLSH
from src.retrieval import HybridRetriever
from src.tune_parameters import deep_size, save_csv

LOG = logging.getLogger(__name__)


def load_frozen_config(config_path, phase4_manifest_path):
    config_path = Path(config_path)
    manifest = json.loads(Path(phase4_manifest_path).read_text(encoding="utf-8"))
    expected = manifest["outputs"][str(config_path)]["sha256"]
    if checksum(config_path) != expected:
        raise ValueError("Frozen config differs from Phase 4 checksum")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = manifest["selection"]["selected"]
    if config["retrieval"] != {key: selected[key] for key in ("tau", "k", "L")} or config["seed"] != selected["seed"]:
        raise ValueError("Parameters differ from Phase 4 selection")
    if config["selection"]["split"] != "validation" or config["selection"]["phase"] != 4:
        raise ValueError("Parameters were not selected on validation")
    return config, expected


def exact_index_bytes(index):
    # Text objects are shared inputs, while tuple/dict/list/ID references are index data.
    return deep_size(index._texts, index._buckets) - deep_size(*index._texts)


def hybrid_index_bytes(hybrid, lsh):
    exact = hybrid._exact
    return deep_size(exact._texts, exact._buckets, lsh._planes, lsh._tables) - deep_size(*exact._texts)


def result_details(method, raw_result, corpus_size):
    if method == "Brute-force":
        return {"ids": raw_result.tolist(), "candidate_ids": None, "candidate_scope": "all_corpus",
                "candidate_count": corpus_size, "comparisons": corpus_size, "route": "brute_force", "count_basis": "full_scan_N"}
    if method == "Exact Hash Only":
        return {"ids": list(raw_result), "candidate_ids": list(raw_result), "candidate_scope": "exact_ids",
                "candidate_count": len(raw_result), "comparisons": 0, "route": "exact_only", "count_basis": "no_vector_comparisons"}
    if method != "Hash + LSH":
        raise ValueError("Unknown method")
    return {"ids": raw_result.ids.tolist(), "candidate_ids": raw_result.candidate_ids.tolist(), "candidate_scope": raw_result.route,
            "candidate_count": len(raw_result.candidate_ids), "comparisons": raw_result.stats["comparisons"],
            "route": raw_result.route, "count_basis": "candidate_count" if raw_result.route == "lsh" else "exact_early_return"}


def verify_result(method, result, query, truth, expected_exact):
    ids = result["ids"]
    if ids != sorted(set(ids)):
        raise AssertionError("Duplicated or unsorted result IDs")
    if method == "Brute-force":
        if ids != truth:
            raise AssertionError("Brute-force differs from Ground Truth")
    elif method == "Exact Hash Only":
        if ids != expected_exact or result["comparisons"] != 0:
            raise AssertionError("Exact-only result/comparison count incorrect")
    elif expected_exact:
        if result["route"] != "exact" or ids != expected_exact or result["comparisons"] != 0:
            raise AssertionError("Hybrid exact-first route incorrect")
    else:
        candidates = result["candidate_ids"]
        if result["route"] != "lsh" or candidates != sorted(set(candidates)):
            raise AssertionError("Invalid LSH candidate IDs/route")
        if ids != sorted(set(candidates) & set(truth)) or result["comparisons"] != len(candidates):
            raise AssertionError("Hybrid semantic result differs from C intersection G")


def run(config_path, measurement_path, phase4_manifest_path, results):
    config, locked_digest = load_frozen_config(config_path, phase4_manifest_path)
    measurement = yaml.safe_load(Path(measurement_path).read_text(encoding="utf-8"))
    if any(not isinstance(measurement[key], int) or measurement[key] < 1 for key in ("warmup_rounds", "repeats", "threads")):
        raise ValueError("Warmup, repeats and threads must be positive integers")
    root, source = Path(results), Path(config["output"])
    raw, figures = root / "raw", root / "figures"
    raw.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    output_names = ("benchmark_plan.json", "test_queries.csv", "test_per_query.csv", "test_timings.csv", "test_results.jsonl",
                    "test_ground_truth.jsonl", "test_scores.npy", "method_summary.csv", "environment.json", "phase5_manifest.json")
    if any((raw / name).exists() for name in output_names) or (figures / "method_comparison.png").exists():
        raise FileExistsError("Phase 5 outputs exist; choose a new --results-dir")
    started = datetime.now(timezone.utc).isoformat()
    save_json(raw / "benchmark_plan.json", {"phase": 5, "started_utc": started, "frozen_config_sha256": locked_digest,
              "parameters": {**config["retrieval"], "seed": config["seed"]}, "measurement": measurement,
              "scope": "test only; precomputed embeddings; interleaved seeded query/method order"})
    phase1_verification = verify_saved(source)
    phase1_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if config["embedding"]["revision"] != phase1_manifest["model_revision"] or config["dataset"]["revision"] != phase1_manifest["dataset_revision"]:
        raise ValueError("Frozen model/dataset revision differs from actual embeddings")
    corpus = load_jsonl(source / "corpus.jsonl")
    queries = load_jsonl(source / "test_queries.jsonl")
    X = np.load(source / "corpus_embeddings.npy", allow_pickle=False)
    Q = np.load(source / "test_queries_embeddings.npy", allow_pickle=False)
    if len(corpus) < 10000 or sum(q["kind"] == "semantic" for q in queries) < 100:
        raise ValueError("Final benchmark requires 10,000 corpus texts and at least 100 semantic test queries")
    texts = [r["text"] for r in corpus]
    tau, k, L = (config["retrieval"][key] for key in ("tau", "k", "L"))
    LOG.info("Frozen test benchmark: N=%d Q=%d tau=%s k=%d L=%d seed=%d", len(X), len(Q), tau, k, L, config["seed"])
    with threadpool_limits(limits=measurement["threads"]):
        tick = perf_counter()
        brute = BruteForceSearch(X)
        brute_build = perf_counter() - tick
        tick = perf_counter()
        exact = ExactHashIndex(texts)
        exact_build = perf_counter() - tick
        tick = perf_counter()
        lsh = RandomHyperplaneLSH(k, L, config["seed"]).fit(X)
        hybrid = HybridRetriever(texts, lsh)
        hybrid_build = perf_counter() - tick
        resources = {"Brute-force": {"index_bytes": 0, "build_seconds": brute_build},
                     "Exact Hash Only": {"index_bytes": exact_index_bytes(exact), "build_seconds": exact_build},
                     "Hash + LSH": {"index_bytes": hybrid_index_bytes(hybrid, lsh), "build_seconds": hybrid_build}}
        # Entire independent Ground Truth is computed before any timed calls.
        scores = np.stack([brute.scores(q) for q in Q])
        truths = [np.flatnonzero(row >= tau).tolist() for row in scores]
        expected_exact = [[i for i, text in enumerate(texts) if text == query["text"]] for query in queries]
        np.save(raw / "test_scores.npy", scores, allow_pickle=False)
        gt_records = [{"query_id": q["query_id"], "kind": q["kind"], "tau": tau, "ground_truth_ids": truth,
                       "ground_truth_scores": score[truth].tolist()} for q, truth, score in zip(queries, truths, scores, strict=True)]
        (raw / "test_ground_truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gt_records), encoding="utf-8")
        save_csv(raw / "test_queries.csv", [{"query_id": q["query_id"], "kind": q["kind"], "ground_truth_count": len(truth),
                 "exact_hit": bool(ids), "exact_ids": json.dumps(ids)} for q, truth, ids in zip(queries, truths, expected_exact, strict=True)])
        callbacks = {"Brute-force": lambda text, q: brute.search(q, tau),
                     "Exact Hash Only": lambda text, q: exact.search(text),
                     "Hash + LSH": lambda text, q: hybrid.search(text, q, tau)}
        reference = {}
        for method in METHODS:
            for i, (query, q) in enumerate(zip(queries, Q, strict=True)):
                result = result_details(method, callbacks[method](query["text"], q), len(X))
                verify_result(method, result, query, truths[i], expected_exact[i])
                reference[method, i] = result
        rng = np.random.default_rng(config["seed"])
        timings = []
        for round_id in range(-measurement["warmup_rounds"], measurement["repeats"]):
            for i in rng.permutation(len(queries)):
                query, q = queries[i], Q[i]
                for method in rng.permutation(METHODS):
                    callback = callbacks[method]
                    if round_id < 0:
                        callback(query["text"], q)
                        continue
                    tick = perf_counter()
                    native = callback(query["text"], q)
                    milliseconds = (perf_counter() - tick) * 1000
                    result = result_details(method, native, len(X))
                    if result != reference[method, i]:
                        raise AssertionError("Timed query result/candidates changed")
                    timings.append({"sequence_id": len(timings), "repeat": round_id, "method": str(method),
                                    "query_id": query["query_id"], "kind": query["kind"], "time_ms": milliseconds,
                                    "candidate_count": result["candidate_count"], "comparisons": result["comparisons"]})
            if round_id >= 0:
                LOG.info("Measured round %d/%d completed", round_id + 1, measurement["repeats"])
        per_query, result_records = [], []
        for method in METHODS:
            for i, query in enumerate(queries):
                result = reference[method, i]
                samples = [r["time_ms"] for r in timings if r["method"] == method and r["query_id"] == query["query_id"]]
                row = {"method": method, "query_id": query["query_id"], "kind": query["kind"], "route": result["route"],
                       **query_metrics(result["ids"], truths[i], result["candidate_count"], len(X), result["comparisons"]),
                       "time_ms": float(np.mean(samples)), "p50_time_ms": float(np.percentile(samples, 50)),
                       "p95_time_ms": float(np.percentile(samples, 95)), "repeats": len(samples), "count_basis": result["count_basis"]}
                per_query.append(row)
                result_records.append({"method": method, "query_id": query["query_id"], "kind": query["kind"], **result})
        summary = summarize(per_query, timings, resources)
        save_csv(raw / "test_timings.csv", timings)
        save_csv(raw / "test_per_query.csv", per_query)
        save_csv(raw / "method_summary.csv", summary)
        (raw / "test_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result_records), encoding="utf-8")
        pools = threadpool_info()
    plot_results(summary, figures, config["retrieval"])
    if checksum(config_path) != locked_digest:
        raise AssertionError("Frozen configuration changed during test benchmark")
    shared = {"raw_corpus_vectors_bytes": X.nbytes, "raw_test_vectors_bytes": Q.nbytes,
              "corpus_jsonl_bytes": (source / "corpus.jsonl").stat().st_size,
              "test_queries_jsonl_bytes": (source / "test_queries.jsonl").stat().st_size,
              "normalized_corpus_workspace_bytes_by_method": {"Brute-force": brute._vectors.nbytes, "Exact Hash Only": 0, "Hash + LSH": lsh._vectors.nbytes},
              "actual_vector_workspace_bytes_in_benchmark_process": brute._vectors.nbytes + lsh._vectors.nbytes,
              "lsh_component_index_bytes": deep_size(lsh._planes, lsh._tables)}
    save_json(raw / "environment.json", {"phase": 5, "split": "test", "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "frozen_config": config,
              "config_sha256_before": locked_digest, "config_sha256_after": checksum(config_path), "test_evaluation_run": True,
              "measurement": measurement, "threadpools": pools, "python": sys.version, "platform": platform.platform(),
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "phase1_verification": phase1_verification, "resources": resources, "shared_storage": shared,
              "memory_definition": "Unique reachable object sizes; planes, buckets, keys, ID containers/integers. Exact raw string objects excluded as shared inputs. Corpus working matrices reported separately. Not RSS or peak memory.",
              "timing_definition": "External perf_counter around each method public API, native result conversion/verification excluded; precomputed embeddings; warmups excluded; group percentiles use all raw repeated samples.",
              "count_definition": "Brute-force=N vector comparisons; exact-only=0; hybrid=deduplicated candidate count on semantic route and 0 on exact route; algorithm-derived counts, not CPU instructions."})
    inputs = [Path(config_path), Path(measurement_path), Path(phase4_manifest_path)]
    inputs += [source / name for name in ("manifest.json", "corpus.jsonl", "corpus_embeddings.npy", "test_queries.jsonl", "test_queries_embeddings.npy")]
    output_paths = [raw / name for name in output_names if name != "phase5_manifest.json"]
    output_paths += [figures / name for name in ("method_comparison.png", "query_group_comparison.png")]
    code = [Path(__file__), Path("src/benchmark_metrics.py"), Path("src/plot_benchmark.py"), Path("src/ground_truth.py"), Path("src/exact_hash.py"), Path("src/random_hyperplane_lsh.py"), Path("src/retrieval.py"), Path("src/metrics.py"), Path("src/tune_parameters.py")]
    save_json(raw / "phase5_manifest.json", {"phase": 5, "split": "test", "corpus_size": len(X), "query_count": len(Q),
              "frozen_config_sha256": locked_digest, "measurement": measurement,
              "checks": {name: "passed" for name in ("frozen_config", "full_ground_truth", "exact_matches", "C_intersection_G", "repeated_results", "candidate_comparison_counts")},
              "inputs": {str(p): checksum(p) for p in inputs}, "code_sha256": {str(p): checksum(p) for p in code},
              "outputs": {str(p): {"bytes": p.stat().st_size, "sha256": checksum(p)} for p in output_paths}})
    for row in summary:
        if row["group"] == "semantic":
            LOG.info("Semantic %s: Recall=%s candidate=%.4f%% time=%.6f ms comparisons=%.2f index=%.3f MiB", row["method"], row["recall"], row["mean_candidate_ratio"] * 100, row["mean_time_ms"], row["mean_comparisons"], row["index_mib"])
    LOG.info("Phase 5 completed: %d queries, %d per-query rows, %d timing samples", len(Q), len(per_query), len(timings))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selected_experiment.yaml")
    parser.add_argument("--measurement", default="configs/benchmark.yaml")
    parser.add_argument("--phase4-manifest", default="results/raw/phase4_manifest.json")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    raw = Path(args.results_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(raw / "phase5.log", encoding="utf-8")])
    try:
        run(args.config, args.measurement, args.phase4_manifest, args.results_dir)
    except Exception:
        LOG.exception("Phase 5 failed")
        raise


if __name__ == "__main__":
    main()
