"""Phase 3 correctness run on validation data; no grid search or final benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import platform
import sys

import numpy as np
import yaml

from src.ground_truth import BruteForceSearch, load_jsonl, threshold_ids
from src.prepare_data import checksum, save_json, verify_saved
from src.random_hyperplane_lsh import RandomHyperplaneLSH, candidate_probability
from src.retrieval import HybridRetriever

LOG = logging.getLogger(__name__)


def run_validation(config_path, k, L, tau, output, ground_truth_dir):
    threshold_ids(np.empty(0), tau)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source, output, gt_dir = Path(config["output"]), Path(output), Path(ground_truth_dir)
    verify_saved(source)
    stem = f"validation_tau_{tau:g}"
    gt_manifest_path = gt_dir / f"{stem}_manifest.json"
    gt_manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
    if gt_manifest["tau"] != tau or gt_manifest["split"] != "validation":
        raise ValueError("Phase 2 Ground Truth threshold/split mismatch")
    for filename, digest in gt_manifest["inputs"].items():
        if checksum(source / filename) != digest:
            raise ValueError(f"Phase 2 source checksum mismatch: {filename}")
    for filename, info in gt_manifest["outputs"].items():
        if checksum(gt_dir / filename) != info["sha256"]:
            raise ValueError(f"Phase 2 output checksum mismatch: {filename}")
    truth = load_jsonl(gt_dir / f"{stem}.jsonl")
    queries = load_jsonl(source / "validation_queries.jsonl")
    corpus = load_jsonl(source / "corpus.jsonl")
    X = np.load(source / "corpus_embeddings.npy", allow_pickle=False)
    Q = np.load(source / "validation_queries_embeddings.npy", allow_pickle=False)
    if len(queries) != len(truth):
        raise ValueError("Ground Truth query count mismatch")
    names = ("hyperplanes.npy", "tables.json", "bucket_stats.json", "queries.jsonl", "summary.json")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in names):
        raise FileExistsError("Phase 3 artifacts exist; select another --output directory")
    LOG.info("Building one provisional index: N=%d k=%d L=%d seed=%d", len(X), k, L, config["seed"])
    index = RandomHyperplaneLSH(k, L, config["seed"]).fit(X)
    reproduced = RandomHyperplaneLSH(k, L, config["seed"]).fit(X)
    if not np.array_equal(index.hyperplanes, reproduced.hyperplanes) or index.tables != reproduced.tables:
        raise AssertionError("Same-seed independent fit is not reproducible")
    del reproduced
    for table in index.tables:
        all_ids = [doc_id for ids in table.values() for doc_id in ids]
        if sorted(all_ids) != list(range(len(X))):
            raise AssertionError("A table did not index each corpus row exactly once")
    hybrid = HybridRetriever((r["text"] for r in corpus), index)
    oracle = BruteForceSearch(X)  # Used only by the validation harness, never by LSH search.
    records = []
    for item, query_vector, expected in zip(queries, Q, truth, strict=True):
        if item["query_id"] != expected["query_id"] or item["kind"] != expected["kind"]:
            raise ValueError("Ground Truth query alignment mismatch")
        direct = index.search_with_stats(query_vector, tau)
        repeated = index.search_with_stats(query_vector, tau)
        if not np.array_equal(direct.ids, repeated.ids) or not np.array_equal(direct.candidate_ids, repeated.candidate_ids):
            raise AssertionError("Repeated LSH query differs")
        candidate_ids = direct.candidate_ids.tolist()
        if candidate_ids != sorted(set(candidate_ids)) or not set(candidate_ids) <= set(range(len(X))):
            raise AssertionError("Candidate IDs invalid or duplicated")
        # Full scans belong to this independent audit, after LSH has already returned.
        fresh_truth = oracle.search(query_vector, tau).tolist()
        if fresh_truth != expected["ground_truth_ids"]:
            raise AssertionError("Current full-scan oracle differs from saved Phase 2 Ground Truth")
        intersection = sorted(set(candidate_ids) & set(fresh_truth))
        if direct.ids.tolist() != intersection or direct.stats["comparisons"] != len(candidate_ids):
            raise AssertionError("Result differs from C(q) intersection G(q), or comparison count is wrong")
        if not np.allclose(direct.scores, oracle.scores(query_vector)[direct.ids], atol=0, rtol=0):
            raise AssertionError("Candidate-only scores differ from current full-scan scores")
        result = hybrid.search(item["text"], query_vector, tau)
        if expected["exact_ids"]:
            if result.route != "exact" or result.ids.tolist() != expected["exact_ids"] or result.stats["comparisons"] != 0:
                raise AssertionError("Exact-first path failed")
        elif result.route != "lsh" or not np.array_equal(result.ids, direct.ids):
            raise AssertionError("Hybrid semantic path differs from standalone LSH")
        records.append({"query_id": item["query_id"], "kind": item["kind"], "candidate_ids": candidate_ids,
                        "lsh_ids": direct.ids.tolist(), "lsh_scores": direct.scores.tolist(),
                        "ground_truth_ids": fresh_truth, "lsh_stats": direct.stats,
                        "hybrid_route": result.route, "hybrid_ids": result.ids.tolist(), "hybrid_stats": result.stats})
    np.save(output / "hyperplanes.npy", index.hyperplanes, allow_pickle=False)
    tables = [{str(signature): list(ids) for signature, ids in table.items()} for table in index.tables]
    save_json(output / "tables.json", tables)
    save_json(output / "bucket_stats.json", index.bucket_stats())
    (output / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    if not np.array_equal(np.load(output / "hyperplanes.npy", allow_pickle=False), index.hyperplanes):
        raise AssertionError("Saved hyperplanes changed")
    if json.loads((output / "tables.json").read_text(encoding="utf-8")) != tables or load_jsonl(output / "queries.jsonl") != records:
        raise AssertionError("Saved tables/query records changed")
    summary = {"phase": 3, "purpose": "single-configuration correctness validation; parameters not calibrated",
               "configuration": {"k": k, "L": L, "tau": tau, "seed": config["seed"]},
               "split": "validation", "corpus_size": len(X), "query_count": len(queries), "dimension": X.shape[1],
               "fit_stats": index.fit_stats, "total_lsh_candidates": sum(len(r["candidate_ids"]) for r in records),
               "total_lsh_comparisons": sum(r["lsh_stats"]["comparisons"] for r in records),
               "exact_routes": sum(r["hybrid_route"] == "exact" for r in records),
               "semantic_routes": sum(r["hybrid_route"] == "lsh" for r in records),
               "theoretical_candidate_probability_at_tau": candidate_probability(tau, k, L),
               "checks": {name: "passed" for name in ("table_membership", "same_seed_fit", "repeated_queries", "candidate_ids",
                          "C_intersection_G", "no_false_positives", "candidate_comparison_counts", "exact_first", "persistence")},
               "score_definition": "float64 normalized row-wise einsum cosine, clipped to [-1,1], inclusive >= tau",
               "timing_scope": "diagnostic single run; per-query times exclude oracle checks and repeated runs; not a benchmark",
               "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
               "inputs": {str(gt_manifest_path): checksum(gt_manifest_path), **{str(source / f): digest for f, digest in gt_manifest["inputs"].items()}},
               "code_sha256": {str(p): checksum(p) for p in (Path(__file__), Path("src/random_hyperplane_lsh.py"), Path("src/retrieval.py"), Path("src/ground_truth.py"))},
               "outputs": {name: {"bytes": (output / name).stat().st_size, "sha256": checksum(output / name)} for name in names if name != "summary.json"}}
    save_json(output / "summary.json", summary)
    LOG.info("All Phase 3 correctness checks passed: %s", json.dumps({k: summary[k] for k in ("query_count", "total_lsh_candidates", "total_lsh_comparisons", "exact_routes", "semantic_routes", "fit_stats")}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--output", default="data/lsh_validation")
    parser.add_argument("--ground-truth-dir", default="data/ground_truth")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "phase3.log", encoding="utf-8")])
    try:
        run_validation(args.config, args.k, args.L, args.tau, output, args.ground_truth_dir)
    except Exception:
        LOG.exception("Phase 3 correctness validation failed")
        raise


if __name__ == "__main__":
    main()
