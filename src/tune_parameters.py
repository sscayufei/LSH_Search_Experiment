"""Validation-only Phase 4 threshold analysis, LSH grid and parameter freeze."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from src.ground_truth import BruteForceSearch, load_jsonl
from src.metrics import recall
from src.prepare_data import checksum, save_json
from src.random_hyperplane_lsh import RandomHyperplaneLSH, candidate_probability

LOG = logging.getLogger(__name__)


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot save an empty measurement table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_validation(source):
    """Read only manifest, corpus and validation files; never open test data."""
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = ("corpus.jsonl", "corpus_embeddings.npy", "validation_queries.jsonl", "validation_queries_embeddings.npy")
    for name in names:
        if checksum(source / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Input checksum mismatch: {name}")
    corpus = load_jsonl(source / names[0])
    all_queries = load_jsonl(source / names[2])
    X = np.load(source / names[1], allow_pickle=False)
    all_Q = np.load(source / names[3], allow_pickle=False)
    if [r["doc_id"] for r in corpus] != list(range(len(corpus))) or len({r["text"] for r in corpus}) != len(corpus):
        raise ValueError("Corpus IDs/texts invalid")
    for records, vectors in ((corpus, X), (all_queries, all_Q)):
        if vectors.dtype != np.float32 or vectors.shape != (len(records), manifest["config"]["embedding"]["dimension"]):
            raise ValueError("Invalid persisted vector shape/dtype")
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Invalid persisted vector values/norms")
    indices = [i for i, item in enumerate(all_queries) if item["kind"] == "semantic"]
    queries = [all_queries[i] for i in indices]
    if not queries or len({r["text"] for r in queries}) != len(queries) or {r["text"] for r in queries} & {r["text"] for r in corpus}:
        raise ValueError("Semantic validation queries empty, duplicated or exactly matched")
    inputs = {str(source / name): checksum(source / name) for name in ("manifest.json", *names)}
    return corpus, queries, X, all_Q[indices], inputs


def quantiles(values):
    return {f"p{int(p * 100):02d}": float(np.quantile(values, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}


def choose_threshold(rows, policy):
    feasible = [r for r in rows if r["source_query_coverage"] >= policy["min_source_query_coverage"]
                and r["random_reference_exceedance"] <= policy["max_random_reference_exceedance"]]
    if feasible:
        return max(feasible, key=lambda r: r["tau"]), True
    return max(rows, key=lambda r: (r["source_query_coverage"], -r["random_reference_exceedance"], r["tau"])), False


def analyze_thresholds(queries, scores, thresholds, policy, seed):
    rng = np.random.default_rng(seed)
    references, source_maxima, source_scores, random_scores = [], [], [], []
    for row, query in enumerate(queries):
        source_ids = sorted({s["corpus_id"] for s in query["sources"]})
        if not source_ids or any(i < 0 or i >= scores.shape[1] for i in source_ids):
            raise ValueError("Invalid semantic query provenance")
        paired = scores[row, source_ids]
        source_scores.extend(paired.tolist())
        source_maxima.append(float(paired.max()))
        allowed = np.setdiff1d(np.arange(scores.shape[1]), source_ids)
        count = policy["random_references_per_query"]
        if count <= 0 or count > len(allowed):
            raise ValueError("Not enough non-source references for requested sample")
        sampled = rng.choice(allowed, size=count, replace=False)
        random_scores.extend(scores[row, sampled].tolist())
        for kind, ids in (("source_pair", source_ids), ("random_non_source", sampled)):
            references.extend({"query_id": query["query_id"], "kind": kind, "corpus_id": int(i), "score": float(scores[row, i])} for i in ids)
    source_maxima, random_scores = np.asarray(source_maxima), np.asarray(random_scores)
    rows = []
    for tau in thresholds:
        if not -1 <= tau <= 1:
            raise ValueError("Threshold outside [-1,1]")
        nonempty = int(np.count_nonzero(np.any(scores >= tau, axis=1)))
        rows.append({"tau": tau, "source_query_coverage": float(np.mean(source_maxima >= tau)),
                     "random_reference_exceedance": float(np.mean(random_scores >= tau)),
                     "queries": len(queries), "nonempty_ground_truth_queries": nonempty,
                     "empty_ground_truth_queries": len(queries) - nonempty})
    choice, feasible = choose_threshold(rows, policy)
    analysis = {"threshold_rows": rows, "selected_tau": choice["tau"], "threshold_policy_satisfied": feasible,
                "source_pair_count": len(source_scores), "random_reference_count": len(random_scores),
                "source_pair_quantiles": quantiles(source_scores), "source_query_max_quantiles": quantiles(source_maxima),
                "random_non_source_quantiles": quantiles(random_scores),
                "interpretation": "Random non-source pairs are a reference distribution, not labeled unrelated pairs or a true false-positive rate. Quora pair labels are not cosine Ground Truth."}
    return analysis, references


def min_tables_for_target(similarity, k, target):
    if not 0 < target < 1:
        raise ValueError("Probability target must be in (0,1)")
    probability = candidate_probability(similarity, k, 1)
    if probability == 0:
        return None
    if probability == 1:
        return 1
    return math.ceil(math.log1p(-target) / math.log1p(-probability))


def deep_size(*roots):
    """Unique reachable Python object sizes; owning ndarray includes its buffer."""
    seen = set()

    def visit(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(visit(k) + visit(v) for k, v in value.items())
        elif isinstance(value, (tuple, list)):
            total += sum(visit(v) for v in value)
        elif isinstance(value, np.ndarray) and value.base is not None:
            total += visit(value.base)
        return total

    return sum(visit(root) for root in roots)


def pareto_front(rows):
    return [i for i, row in enumerate(rows) if not any(
        other["recall"] >= row["recall"] and other["mean_candidate_ratio"] <= row["mean_candidate_ratio"]
        and (other["recall"] > row["recall"] or other["mean_candidate_ratio"] < row["mean_candidate_ratio"])
        for other in rows)]


def select_configuration(rows, policy):
    limits = (("mean_candidate_ratio", "max_mean_candidate_ratio"), ("p95_candidate_ratio", "max_p95_candidate_ratio"), ("index_bytes", "max_index_bytes"))
    feasible = [r for r in rows if all(r[key] <= policy[cap] for key, cap in limits)]
    order = lambda r: (-r["recall"], r["mean_candidate_ratio"], r["L"], r["k"])
    if feasible:
        return min(feasible, key=order), True
    frontier = [rows[i] for i in pareto_front(rows)]
    violation = lambda r: sum(max(0, r[key] / policy[cap] - 1) for key, cap in limits)
    return min(frontier, key=lambda r: (violation(r), *order(r))), False


def evaluate_configuration(X, Q, queries, truths, k, L, tau, seed, measurement):
    index = RandomHyperplaneLSH(k, L, seed).fit(X)
    # Measure only planes and buckets; exclude the shared corpus working matrix.
    index_bytes = deep_size(index._planes, index._tables)
    reference = [index.search_with_stats(q, tau) for q in Q]
    for result, truth in zip(reference, truths, strict=True):
        if result.ids.tolist() != sorted(set(result.candidate_ids) & set(truth)):
            raise AssertionError("LSH result differs from C intersection G")
        if result.stats["comparisons"] != len(result.candidate_ids):
            raise AssertionError("Candidate comparison count differs")
    for _ in range(measurement["warmup_rounds"]):
        for q in Q:
            index.search_with_stats(q, tau)
    timings = []
    per_query_times = [[] for _ in Q]
    for repeat in range(measurement["repeats"]):
        for i, q in enumerate(Q):
            start = perf_counter()
            result = index.search_with_stats(q, tau)
            milliseconds = (perf_counter() - start) * 1000
            if not np.array_equal(result.ids, reference[i].ids) or not np.array_equal(result.candidate_ids, reference[i].candidate_ids):
                raise AssertionError("Repeated query differs")
            per_query_times[i].append(milliseconds)
            timings.append({"k": k, "L": L, "query_id": queries[i]["query_id"], "repeat": repeat,
                            "time_ms": milliseconds, "comparisons": result.stats["comparisons"],
                            "signature_ms": result.stats["signature_seconds"] * 1000,
                            "bucket_lookup_ms": result.stats["bucket_lookup_seconds"] * 1000,
                            "verification_ms": result.stats["verification_seconds"] * 1000})
    per_query = []
    for i, (result, truth) in enumerate(zip(reference, truths, strict=True)):
        per_query.append({"k": k, "L": L, "tau": tau, "query_id": queries[i]["query_id"], "ground_truth_count": len(truth),
                          "candidate_count": len(result.candidate_ids), "result_count": len(result.ids),
                          "recall": recall(result.ids, truth), "candidate_ratio": len(result.candidate_ids) / len(X),
                          "comparisons": result.stats["comparisons"], "mean_time_ms": float(np.mean(per_query_times[i]))})
    defined = [r["recall"] for r in per_query if r["recall"] is not None]
    if not defined:
        raise ValueError("No nonempty Ground Truth; conditional Recall is undefined")
    row = {"k": k, "L": L, "tau": tau, "seed": seed, "queries": len(Q),
           "nonempty_ground_truth_queries": len(defined), "empty_ground_truth_queries": len(Q) - len(defined),
           "recall": float(np.mean(defined)), "mean_candidate_ratio": float(np.mean([r["candidate_ratio"] for r in per_query])),
           "p95_candidate_ratio": float(np.percentile([r["candidate_ratio"] for r in per_query], 95)),
           "mean_candidates": float(np.mean([r["candidate_count"] for r in per_query])),
           "mean_comparisons": float(np.mean([r["comparisons"] for r in per_query])),
           "mean_time_ms": float(np.mean([r["mean_time_ms"] for r in per_query])),
           "p95_time_ms": float(np.percentile([r["mean_time_ms"] for r in per_query], 95)),
           "index_bytes": index_bytes, "build_seconds": index.fit_stats["build_seconds"],
           "signature_build_seconds": index.fit_stats["signature_seconds"], "id_references": len(X) * L}
    return row, per_query, timings


def draw_figures(figures, grid, selected, refs, tau):
    os.environ["MPLCONFIGDIR"] = str((figures.parent / ".matplotlib").resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for k in sorted({r["k"] for r in grid}):
        group = [r for r in grid if r["k"] == k]
        ax.plot([r["mean_candidate_ratio"] * 100 for r in group], [r["recall"] for r in group], "o-", label=f"k={k}")
        for r in group:
            ax.annotate(f"L={r['L']}", (r["mean_candidate_ratio"] * 100, r["recall"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.scatter([selected["mean_candidate_ratio"] * 100], [selected["recall"]], marker="*", s=230, c="black", label="selected", zorder=5)
    front = [grid[i] for i in pareto_front(grid)]
    front.sort(key=lambda r: r["mean_candidate_ratio"])
    ax.plot([r["mean_candidate_ratio"] * 100 for r in front], [r["recall"] for r in front], "k--", alpha=0.4, label="Pareto frontier")
    ax.set(xlabel="Mean candidate ratio (%)", ylabel="Conditional semantic Recall", ylim=(0, 1.06), title=f"Validation only, tau={tau:g}, seed={selected['seed']}")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(figures / "recall_candidate.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for kind, label in (("source_pair", "Source pairs"), ("random_non_source", "Random non-source references")):
        ax.hist([r["score"] for r in refs if r["kind"] == kind], bins=np.linspace(-1, 1, 81), density=True, histtype="step", linewidth=1.8, label=label)
    ax.axvline(tau, color="black", linestyle="--", label=f"selected tau={tau:g}")
    ax.set(xlabel="Cosine similarity", ylabel="Density", title="Threshold evidence: validation only")
    ax.set_xlim(max(-1, min(r["score"] for r in refs) - 0.05), 1)
    ax.legend()
    fig.savefig(figures / "threshold_analysis.png")
    plt.close(fig)


def run(config_path, plan_path, results, lock_file):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    if plan["measurement"]["repeats"] < 1 or plan["measurement"]["warmup_rounds"] < 1:
        raise ValueError("At least one warmup and one measured repeat required")
    results, lock_file = Path(results), Path(lock_file)
    raw, figures = results / "raw", results / "figures"
    raw.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if lock_file.exists() or any((raw / name).exists() for name in ("phase4_plan.json", "validation_grid.csv", "phase4_manifest.json")):
        raise FileExistsError("Phase 4 outputs or lock exist; use a new results directory and lock file")
    save_json(raw / "phase4_plan.json", {"seed": config["seed"], "policy": plan, "scope": "semantic validation queries only", "created_utc": datetime.now(timezone.utc).isoformat()})
    corpus, queries, X, Q, inputs = load_validation(config["output"])
    LOG.info("Loaded only corpus and semantic validation data: N=%d Q=%d", len(corpus), len(queries))
    inputs.update({str(config_path): checksum(config_path), str(plan_path): checksum(plan_path)})
    with threadpool_limits(limits=plan["measurement"]["threads"]):
        oracle = BruteForceSearch(X)
        scores = np.stack([oracle.scores(q) for q in Q])
        analysis, refs = analyze_thresholds(queries, scores, plan["thresholds"], plan["threshold_policy"], config["seed"])
        save_json(raw / "threshold_analysis.json", analysis)
        save_csv(raw / "threshold_reference_pairs.csv", refs)
        save_csv(raw / "threshold_candidates.csv", analysis["threshold_rows"])
        tau = analysis["selected_tau"]
        LOG.info("Selected tau=%s before fitting LSH; policy satisfied=%s", tau, analysis["threshold_policy_satisfied"])
        np.save(raw / "validation_semantic_scores.npy", scores, allow_pickle=False)
        truths = [np.flatnonzero(row >= tau).tolist() for row in scores]
        (raw / "validation_ground_truth.jsonl").write_text("".join(json.dumps({"query_id": q["query_id"], "tau": tau, "ground_truth_ids": truth}) + "\n" for q, truth in zip(queries, truths, strict=True)), encoding="utf-8")
        theory = []
        for k in plan["grid"]["k"]:
            for L in plan["grid"]["L"]:
                theory.append({"k": k, "L": L, "s1": tau, "s0": plan["theory"]["s0"],
                               "P_s1": candidate_probability(tau, k, L), "P_s0": candidate_probability(plan["theory"]["s0"], k, L),
                               "min_L_for_target": min_tables_for_target(tau, k, plan["theory"]["target_boundary_probability"]),
                               "conditional_noise_expectation_at_s0": len(X) * candidate_probability(plan["theory"]["s0"], k, L)})
        save_csv(raw / "theory_grid.csv", theory)
        grid, all_queries, all_timings = [], [], []
        for item in theory:
            row, per_query, timings = evaluate_configuration(X, Q, queries, truths, item["k"], item["L"], tau, config["seed"], plan["measurement"])
            row.update(theory_P_s1=item["P_s1"], theory_P_s0=item["P_s0"])
            grid.append(row)
            all_queries.extend(per_query)
            all_timings.extend(timings)
            LOG.info("k=%d L=%d Recall=%.6f candidate=%.4f%% memory=%.3f MiB", row["k"], row["L"], row["recall"], row["mean_candidate_ratio"] * 100, row["index_bytes"] / 2**20)
        selected, budget_ok = select_configuration(grid, plan["selection"])
        frontier = pareto_front(grid)
        for i, row in enumerate(grid):
            row.update(pareto=i in frontier, selected=row is selected)
        save_csv(raw / "validation_grid.csv", grid)
        save_csv(raw / "validation_per_query.csv", all_queries)
        save_csv(raw / "validation_timings.csv", all_timings)
        selection = {"selected": selected, "budget_satisfied": budget_ok,
                     "measured_recall_goal_met": selected["recall"] >= plan["selection"]["measured_recall_goal"],
                     "theoretical_probability_goal_met": selected["theory_P_s1"] >= plan["theory"]["target_boundary_probability"],
                     "policy": plan["selection"], "threshold_policy_satisfied": analysis["threshold_policy_satisfied"],
                     "recall_definition": "mean over semantic validation queries with nonempty Ground Truth; all queries retained in raw tables",
                     "memory_definition": "recursive sys.getsizeof of unique planes and bucket objects including ID/key integers and array buffers; excludes corpus working copy; not RSS"}
        save_json(raw / "parameter_selection.json", selection)
        pools = threadpool_info()
    draw_figures(figures, grid, selected, refs, tau)
    frozen = dict(config)
    frozen["retrieval"] = {"tau": tau, "k": selected["k"], "L": selected["L"]}
    frozen["selection"] = {"phase": 4, "split": "validation", "frozen_utc": datetime.now(timezone.utc).isoformat(),
                           "test_evaluation_run": False, "grid_sha256": checksum(raw / "validation_grid.csv"),
                           "threshold_analysis_sha256": checksum(raw / "threshold_analysis.json"), "policy_sha256": checksum(plan_path),
                           "budget_satisfied": budget_ok, "validation_recall_goal_met": selection["measured_recall_goal_met"]}
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding="utf-8")
    output_paths = [p for p in raw.iterdir() if p.is_file() and p.name not in ("phase4.log", "phase4_manifest.json")]
    output_paths += list(figures.glob("*.png")) + [lock_file]
    manifest = {"phase": 4, "scope": "validation only; no test query files opened for selection", "inputs": inputs,
                "code_sha256": {str(p): checksum(p) for p in (Path(__file__), Path("src/metrics.py"), Path("src/random_hyperplane_lsh.py"), Path("src/ground_truth.py"))},
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "threadpools": pools,
                "measurement": plan["measurement"], "selection": selection,
                "outputs": {str(p): {"bytes": p.stat().st_size, "sha256": checksum(p)} for p in output_paths}}
    save_json(raw / "phase4_manifest.json", manifest)
    LOG.info("Frozen parameters: tau=%s k=%d L=%d seed=%d budget_satisfied=%s recall_goal_met=%s", tau, selected["k"], selected["L"], config["seed"], budget_ok, selection["measured_recall_goal_met"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--plan", default="configs/phase4.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--lock-file", default="configs/selected_experiment.yaml")
    args = parser.parse_args()
    raw = Path(args.results_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(raw / "phase4.log", encoding="utf-8")])
    try:
        run(args.config, args.plan, args.results_dir, args.lock_file)
    except Exception:
        LOG.exception("Phase 4 failed")
        raise


if __name__ == "__main__":
    main()
