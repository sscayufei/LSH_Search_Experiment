"""Phase 5 aggregates, preserving undefined Recall and query group boundaries."""
import numpy as np

from src.metrics import recall


METHODS = ("Brute-force", "Exact Hash Only", "Hash + LSH")


def query_metrics(ids, truth, candidate_count, corpus_size, comparisons):
    if corpus_size <= 0 or not 0 <= candidate_count <= corpus_size or comparisons < 0:
        raise ValueError("Invalid corpus/candidate/comparison counts")
    return {"ground_truth_count": len(set(truth)), "candidate_count": candidate_count,
            "result_count": len(set(ids)), "recall": recall(ids, truth),
            "candidate_ratio": candidate_count / corpus_size, "comparisons": comparisons}


def summarize(per_query, timings, resources):
    rows = []
    for method in METHODS:
        for group in ("semantic", "exact", "all"):
            subset = [r for r in per_query if r["method"] == method and (group == "all" or r["kind"] == group)]
            if not subset:
                continue
            ids = {r["query_id"] for r in subset}
            samples = [r["time_ms"] for r in timings if r["method"] == method and r["query_id"] in ids]
            if not samples:
                raise ValueError("Missing timing samples")
            defined = [r["recall"] for r in subset if r["recall"] is not None]
            rows.append({"method": method, "group": group, "queries": len(subset),
                         "nonempty_ground_truth_queries": len(defined), "empty_ground_truth_queries": len(subset) - len(defined),
                         "nonempty_ground_truth_ratio": len(defined) / len(subset),
                         "recall": float(np.mean(defined)) if defined else None,
                         "mean_candidate_ratio": float(np.mean([r["candidate_ratio"] for r in subset])),
                         "p95_candidate_ratio": float(np.percentile([r["candidate_ratio"] for r in subset], 95)),
                         "mean_candidates": float(np.mean([r["candidate_count"] for r in subset])),
                         "mean_comparisons": float(np.mean([r["comparisons"] for r in subset])),
                         "mean_time_ms": float(np.mean(samples)), "p50_time_ms": float(np.percentile(samples, 50)),
                         "p95_time_ms": float(np.percentile(samples, 95)), "timing_samples": len(samples),
                         "index_bytes": resources[method]["index_bytes"], "index_mib": resources[method]["index_bytes"] / 2**20,
                         "build_seconds": resources[method]["build_seconds"]})
    return rows
