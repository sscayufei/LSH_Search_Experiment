"""Phase 6: isolated real-size workers and explicitly labeled scale predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from src.benchmark import hybrid_index_bytes, load_frozen_config
from src.ground_truth import BruteForceSearch, load_jsonl
from src.metrics import recall
from src.prepare_data import checksum, save_json
from src.prepare_scale_data import verify_scale_data
from src.random_hyperplane_lsh import RandomHyperplaneLSH, candidate_probability
from src.retrieval import HybridRetriever
from src.tune_parameters import deep_size, save_csv

LOG = logging.getLogger(__name__)


def prediction_rows(measured_baseline, sizes, similarities, k, L, dimension):
    """No measured timing or index-memory values are invented at unrun sizes."""
    if measured_baseline["N"] <= 0 or any(N <= 0 for N in sizes):
        raise ValueError("Prediction sizes must be positive")
    rows = []
    for N in sizes:
        for similarity in similarities:
            probability = candidate_probability(similarity, k, L)
            rows.append({"N": N, "value_type": "conditional_theory", "quantity": "mean_false_candidates",
                         "reference_similarity": similarity, "probability_or_rate": probability,
                         "value": N * probability, "units": "vectors",
                         "assumption": "all background similarities equal reference_similarity; N approximates background count"})
        for quantity, field in (("mean_candidates", "mean_candidates"), ("mean_false_candidates", "mean_false_candidates")):
            fraction = measured_baseline[field] / measured_baseline["N"]
            rows.append({"N": N, "value_type": "extrapolated_from_measured_10k", "quantity": quantity,
                         "reference_similarity": "", "probability_or_rate": fraction, "value": N * fraction, "units": "vectors",
                         "assumption": "10k candidate fraction and similarity distribution remain stationary; not a measurement"})
        for quantity, width in (("float32_vector_payload", 4), ("float64_normalized_workspace", 8)):
            rows.append({"N": N, "value_type": "arithmetic_storage_formula", "quantity": quantity,
                         "reference_similarity": "", "probability_or_rate": "", "value": N * dimension * width, "units": "bytes",
                         "assumption": "dense N by d vector payload only; excludes headers, index, queries and process overhead"})
    return rows


def load_scale_queries(config):
    source = Path(config["output"])
    records = load_jsonl(source / "test_queries.jsonl")
    matrix = np.load(source / "test_queries_embeddings.npy", allow_pickle=False)
    selected = [i for i, row in enumerate(records) if row["kind"] == "semantic"]
    if len(selected) < 100:
        raise ValueError("At least 100 fixed semantic queries required")
    return [records[i] for i in selected], matrix[selected]


def run_worker(config_path, plan_path, phase4_manifest, scale_dir, results_dir, N):
    config, config_digest = load_frozen_config(config_path, phase4_manifest)
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    if N not in plan["measured_sizes"]:
        raise ValueError("Worker size is not in the predeclared measurement plan")
    output = Path(results_dir) / "raw" / f"scale_n{N}"
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("manifest.json", "summary.json", "per_query.csv")):
        raise FileExistsError("Worker outputs exist; choose another results directory")
    data_manifest, records, vectors, max_error = verify_scale_data(scale_dir)
    if N > len(records) or data_manifest["frozen_config_sha256"] != config_digest:
        raise ValueError("Scale corpus size/config differs from frozen plan")
    queries, Q = load_scale_queries(config)
    corpus_texts = {r["text"] for r in records}
    if any(q["text"] in corpus_texts for q in queries):
        raise ValueError("Semantic query text leaked into expanded corpus")
    del corpus_texts
    X = vectors[:N]
    tau, k, L = (config["retrieval"][key] for key in ("tau", "k", "L"))
    LOG.info("N=%d: recomputing full Ground Truth for the same %d semantic queries", N, len(queries))
    with threadpool_limits(limits=plan["threads"]):
        start = perf_counter()
        oracle = BruteForceSearch(X)
        truths = [oracle.search(q, tau).tolist() for q in Q]
        oracle_seconds = perf_counter() - start
        del oracle
        gc.collect()  # Do not retain two N*d float64 work matrices during index construction.
        LOG.info("N=%d: Ground Truth complete in %.3fs; building the unchanged hybrid index", N, oracle_seconds)
        start = perf_counter()
        lsh = RandomHyperplaneLSH(k, L, config["seed"]).fit(X)
        hybrid = HybridRetriever((r["text"] for r in records[:N]), lsh)
        build_seconds = perf_counter() - start
        index_bytes = hybrid_index_bytes(hybrid, lsh)
        lsh_bytes = deep_size(lsh._planes, lsh._tables)
        planes_sha256 = hashlib.sha256(lsh.hyperplanes.tobytes()).hexdigest()
        bucket_rows = lsh.bucket_stats()
        all_bucket_sizes = [len(ids) for table in lsh.tables for ids in table.values()]
        for table in lsh.tables:
            if sorted(i for ids in table.values() for i in ids) != list(range(N)):
                raise AssertionError("A table did not contain every corpus ID exactly once")
        reference = [hybrid.search(q["text"], vector, tau) for q, vector in zip(queries, Q, strict=True)]
        per_query, result_rows = [], []
        for item, result, truth in zip(queries, reference, truths, strict=True):
            candidates = result.candidate_ids.tolist()
            if result.route != "lsh" or result.ids.tolist() != sorted(set(candidates) & set(truth)):
                raise AssertionError("Semantic route differs from candidate intersection Ground Truth")
            if result.stats["comparisons"] != len(candidates):
                raise AssertionError("Comparison count differs from deduplicated candidates")
            per_query.append({"N": N, "value_type": "measured", "query_id": item["query_id"],
                              "ground_truth_count": len(truth), "candidate_count": len(candidates), "result_count": len(result.ids),
                              "false_candidate_count": len(set(candidates) - set(truth)), "recall": recall(result.ids, truth),
                              "candidate_ratio": len(candidates) / N, "comparisons": result.stats["comparisons"]})
            result_rows.append({"query_id": item["query_id"], "candidate_ids": candidates, "result_ids": result.ids.tolist(), "ground_truth_ids": truth})
        rng = np.random.default_rng(config["seed"])
        timings = []
        for repeat in range(-plan["warmup_rounds"], plan["repeats"]):
            for i in rng.permutation(len(queries)):
                item, vector = queries[i], Q[i]
                if repeat < 0:
                    hybrid.search(item["text"], vector, tau)
                    continue
                tick = perf_counter()
                result = hybrid.search(item["text"], vector, tau)
                milliseconds = (perf_counter() - tick) * 1000
                if not np.array_equal(result.ids, reference[i].ids) or not np.array_equal(result.candidate_ids, reference[i].candidate_ids):
                    raise AssertionError("Repeated candidate/results differ")
                timings.append({"N": N, "value_type": "measured", "sequence_id": len(timings), "repeat": repeat,
                                "query_id": item["query_id"], "time_ms": milliseconds, "comparisons": result.stats["comparisons"]})
        for row in per_query:
            samples = [r["time_ms"] for r in timings if r["query_id"] == row["query_id"]]
            row.update(mean_time_ms=float(np.mean(samples)), p95_time_ms=float(np.percentile(samples, 95)))
        defined = [r["recall"] for r in per_query if r["recall"] is not None]
        summary = {"N": N, "value_type": "measured", "queries": len(queries), "nonempty_ground_truth_queries": len(defined),
                   "recall": float(np.mean(defined)) if defined else None,
                   "mean_candidates": float(np.mean([r["candidate_count"] for r in per_query])),
                   "p95_candidates": float(np.percentile([r["candidate_count"] for r in per_query], 95)),
                   "mean_false_candidates": float(np.mean([r["false_candidate_count"] for r in per_query])),
                   "mean_candidate_ratio": float(np.mean([r["candidate_ratio"] for r in per_query])),
                   "p95_candidate_ratio": float(np.percentile([r["candidate_ratio"] for r in per_query], 95)),
                   "mean_comparisons": float(np.mean([r["comparisons"] for r in per_query])),
                   "mean_time_ms": float(np.mean([r["time_ms"] for r in timings])),
                   "p95_time_ms": float(np.percentile([r["time_ms"] for r in timings], 95)),
                   "index_bytes": index_bytes, "lsh_index_bytes": lsh_bytes, "normalized_workspace_bytes": lsh._vectors.nbytes,
                   "build_seconds": build_seconds, "oracle_seconds": oracle_seconds,
                   "max_bucket_size": max(all_bucket_sizes, default=0), "p95_occupied_bucket_size": float(np.percentile(all_bucket_sizes, 95)),
                   "mean_occupied_buckets_per_table": float(np.mean([r["occupied_buckets"] for r in bucket_rows])),
                   "planes_sha256": planes_sha256}
        pools = threadpool_info()
    save_csv(output / "per_query.csv", per_query)
    save_csv(output / "timings.csv", timings)
    (output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result_rows), encoding="utf-8")
    save_json(output / "bucket_stats.json", bucket_rows)
    save_json(output / "summary.json", summary)
    files = ("per_query.csv", "timings.csv", "results.jsonl", "bucket_stats.json", "summary.json")
    save_json(output / "manifest.json", {"phase": 6, "N": N, "value_type": "measured", "parameters": {**config["retrieval"], "seed": config["seed"]},
              "measurement": {key: plan[key] for key in ("warmup_rounds", "repeats", "threads")}, "threadpools": pools,
              "max_vector_norm_error": max_error, "inputs": {str(p): checksum(p) for p in (Path(config_path), Path(plan_path), Path(scale_dir) / "manifest.json", Path(config["output"]) / "test_queries.jsonl", Path(config["output"]) / "test_queries_embeddings.npy")},
              "outputs": {name: {"bytes": (output / name).stat().st_size, "sha256": checksum(output / name)} for name in files}})
    LOG.info("Measured N=%d: mean candidates=%.2f false candidates=%.2f Recall=%s time=%.4fms index=%.3fMiB", N, summary["mean_candidates"], summary["mean_false_candidates"], summary["recall"], summary["mean_time_ms"], index_bytes / 2**20)
    return summary


def run(config_path, plan_path, phase4_manifest, scale_dir, results_dir):
    config, frozen_hash = load_frozen_config(config_path, phase4_manifest)
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    data_manifest, records, vectors, error = verify_scale_data(scale_dir)
    del records, vectors
    gc.collect()
    if plan["optional_improvement"] != "none":
        raise ValueError("Optional improvements require explicit selection; this workflow implements none")
    output = Path(results_dir) / "raw"
    figures = Path(results_dir) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("scale_plan.json", "scale_summary.csv", "scale_manifest.json")):
        raise FileExistsError("Scale results exist; use another --results-dir")
    save_json(output / "scale_plan.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": frozen_hash,
              "plan": plan, "scope": "same 100 semantic test queries, original 10k fixed prefix; independent process for each measured size"})
    summaries = []
    for N in plan["measured_sizes"]:
        LOG.info("Launching isolated worker for N=%d", N)
        command = [sys.executable, "-m", "src.scale_analysis", "--worker-size", str(N), "--config", str(config_path),
                   "--plan", str(plan_path), "--phase4-manifest", str(phase4_manifest), "--scale-dir", str(scale_dir), "--results-dir", str(results_dir)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   env={**os.environ, "PYTHONUTF8": "1"}, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for line in process.stdout:
            LOG.info("[worker %d] %s", N, line.rstrip())
        if process.wait() != 0:
            raise RuntimeError(f"Scale worker N={N} failed; real partial artifacts preserved")
        summary = json.loads((output / f"scale_n{N}" / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
    if len({r["planes_sha256"] for r in summaries}) != 1:
        raise AssertionError("Different scales used different hyperplanes")
    previous = None
    for row in summaries:
        current = load_jsonl(output / f"scale_n{row['N']}" / "results.jsonl")
        if previous:
            for a, b in zip(previous, current, strict=True):
                if a["query_id"] != b["query_id"] or not set(a["candidate_ids"]) <= set(b["candidate_ids"]) or not set(a["ground_truth_ids"]) <= set(b["ground_truth_ids"]):
                    raise AssertionError("Nested prefixes did not preserve candidate/Ground Truth membership")
        previous = current
    save_csv(output / "scale_summary.csv", summaries)
    predictions = prediction_rows(summaries[0], plan["prediction_sizes"], plan["reference_similarities"], config["retrieval"]["k"], config["retrieval"]["L"], config["embedding"]["dimension"])
    save_csv(output / "scale_predictions.csv", predictions)
    from src.plot_scale import plot_scale
    plot_scale(summaries, predictions, figures)
    if checksum(config_path) != frozen_hash:
        raise AssertionError("Frozen parameters changed during scale analysis")
    files = [output / "scale_plan.json", output / "scale_summary.csv", output / "scale_predictions.csv"]
    files += [p for N in plan["measured_sizes"] for p in (output / f"scale_n{N}").iterdir() if p.is_file()]
    files += [figures / "scale_analysis.png", figures / "scale_diagnostics.png"]
    source_paths = [Path(config_path), Path(plan_path), Path(scale_dir) / "manifest.json", Path("results/raw/phase5_manifest.json"), Path("results/raw/scale_resource_preflight.json")]
    code_paths = [Path(__file__), Path("src/plot_scale.py"), Path("src/prepare_scale_data.py"), Path("src/random_hyperplane_lsh.py"), Path("src/ground_truth.py"), Path("src/retrieval.py"), Path("src/benchmark.py"), Path("src/tune_parameters.py")]
    save_json(output / "scale_manifest.json", {"phase": 6, "config_sha256": frozen_hash, "measured_sizes": plan["measured_sizes"],
              "prediction_sizes": plan["prediction_sizes"], "million_size_measured": False, "parameters": {**config["retrieval"], "seed": config["seed"]},
              "checks": {name: "passed" for name in ("real_embeddings", "same_hyperplanes", "same_queries", "nested_candidates", "nested_ground_truth", "C_intersection_G", "frozen_parameters")},
              "distribution_caveat": data_manifest["distribution_caveat"], "python": sys.version, "platform": platform.platform(),
              "measurement_scope": "repeated precomputed-query hybrid API; GT, build, memory traversal and embedding excluded; index memory is a unique-object estimate",
              "inputs": {str(p): checksum(p) for p in source_paths}, "code_sha256": {str(p): checksum(p) for p in code_paths},
              "outputs": {str(p): {"bytes": p.stat().st_size, "sha256": checksum(p)} for p in files}})
    LOG.info("Phase 6 complete: measured %s; million-scale rows are predictions only", plan["measured_sizes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selected_experiment.yaml")
    parser.add_argument("--plan", default="configs/scale.yaml")
    parser.add_argument("--phase4-manifest", default="results/raw/phase4_manifest.json")
    parser.add_argument("--scale-dir", default="data/scale")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--worker-size", type=int)
    args = parser.parse_args()
    raw = Path(args.results_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(raw / "scale_analysis.log", encoding="utf-8")])
    try:
        if args.worker_size is not None:
            run_worker(args.config, args.plan, args.phase4_manifest, args.scale_dir, args.results_dir, args.worker_size)
        else:
            run(args.config, args.plan, args.phase4_manifest, args.scale_dir, args.results_dir)
    except Exception:
        LOG.exception("Phase 6 scale analysis failed")
        raise


if __name__ == "__main__":
    main()
