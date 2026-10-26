"""Independent full-corpus cosine oracle and Phase 2 artifact generation."""
from __future__ import annotations

import argparse
import json
import logging
from numbers import Real
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

from src.exact_hash import ExactHashIndex
from src.prepare_data import checksum, save_json, verify_saved

LOG = logging.getLogger(__name__)


def normalize_rows(vectors):
    """Float64 working copy; reject undefined cosine for zero or invalid rows."""
    raw = np.asarray(vectors)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] == 0:
        raise ValueError("Expected a real numeric matrix with positive dimension")
    values = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Vectors must be finite")
    # Scale first to avoid overflow/underflow in norms for general valid inputs.
    scales = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError("Zero vectors have undefined cosine similarity")
    values /= scales
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def threshold_ids(scores, tau):
    """Inclusive >= threshold; ascending row IDs, no top-k truncation."""
    if not isinstance(tau, Real) or not np.isfinite(tau) or not -1 <= tau <= 1:
        raise ValueError("tau must be finite and in [-1, 1]")
    scores = np.asarray(scores)
    if scores.dtype.kind not in "fiu" or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite one-dimensional scores")
    return np.flatnonzero(scores >= tau)


def cosine_scores_normalized(vectors, query):
    """Same row-wise reduction for full-corpus and candidate-only comparisons.

    Inputs are already normalized float64 vectors. Keeping the reduction independent
    of row count avoids different BLAS kernels changing threshold boundary rounding.
    """
    return np.clip(np.einsum("nd,d->n", vectors, query, optimize=False), -1.0, 1.0)


class BruteForceSearch:
    """Normalize once, then compute cosine against every corpus row per query."""

    def __init__(self, vectors):
        self._vectors = normalize_rows(vectors)
        self._vectors.flags.writeable = False

    def scores(self, query):
        query = np.asarray(query)
        if query.ndim != 1 or query.shape[0] != self._vectors.shape[1]:
            raise ValueError("Query must be a vector matching corpus dimension")
        normalized = normalize_rows(query[None, :])[0]
        # Roundoff can place a normalized dot product just outside [-1, 1].
        return cosine_scores_normalized(self._vectors, normalized)

    def search(self, query, tau):
        return threshold_ids(self.scores(query), tau)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def generate(config_path, split, tau, destination):
    threshold_ids(np.empty(0), tau)  # Validate before loading or writing artifacts.
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source = Path(config["output"])
    verification = verify_saved(source)
    corpus = load_jsonl(source / "corpus.jsonl")
    query_file = source / f"{split}_queries.jsonl"
    queries = load_jsonl(query_file)
    vectors = np.load(source / "corpus_embeddings.npy", allow_pickle=False)
    query_vectors = np.load(source / f"{split}_queries_embeddings.npy", allow_pickle=False)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_tau_{tau:g}"
    result_path = destination / f"{stem}.jsonl"
    scores_path = destination / f"{stem}_scores.npy"
    manifest_path = destination / f"{stem}_manifest.json"
    if any(p.exists() for p in (result_path, scores_path, manifest_path)):
        raise FileExistsError("Ground Truth output exists; choose a new --output directory")
    exact = ExactHashIndex(r["text"] for r in corpus)
    brute = BruteForceSearch(vectors)
    all_scores = []
    records = []
    for item, query_vector in zip(queries, query_vectors, strict=True):
        scores = brute.scores(query_vector)
        ids = threshold_ids(scores, tau)
        repeated = brute.search(query_vector, tau)
        if not np.array_equal(ids, repeated):
            raise AssertionError("Repeated Ground Truth computation differs")
        exact_ids = exact.search(item["text"])
        if item["kind"] == "exact" and exact_ids != [item["corpus_id"]]:
            raise AssertionError("Exact query failed to match its original corpus row")
        if item["kind"] == "semantic" and exact_ids:
            raise AssertionError("Semantic query unexpectedly matched an original string")
        records.append({"query_id": item["query_id"], "kind": item["kind"], "tau": tau,
                        "exact_ids": exact_ids, "ground_truth_ids": ids.tolist(),
                        "ground_truth_scores": scores[ids].tolist()})
        all_scores.append(scores)
    score_matrix = np.stack(all_scores) if all_scores else np.empty((0, len(corpus)), dtype=np.float64)
    result_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    np.save(scores_path, score_matrix, allow_pickle=False)
    # Independently reload persisted data and recompute score threshold membership.
    saved_scores = np.load(scores_path, allow_pickle=False)
    saved_records = load_jsonl(result_path)
    if len(saved_records) != len(queries) or not np.array_equal(saved_scores, score_matrix):
        raise AssertionError("Ground Truth serialization round trip failed")
    for record, scores in zip(saved_records, saved_scores, strict=True):
        expected = [i for i, score in enumerate(scores) if score >= tau]
        if record["ground_truth_ids"] != expected or record["ground_truth_scores"] != scores[expected].tolist():
            raise AssertionError("Persisted Ground Truth disagrees with full scores")
    counts = {kind: {"queries": sum(r["kind"] == kind for r in records),
                     "exact_hits": sum(r["kind"] == kind and bool(r["exact_ids"]) for r in records),
                     "nonempty_ground_truth": sum(r["kind"] == kind and bool(r["ground_truth_ids"]) for r in records)}
              for kind in ("semantic", "exact")}
    inputs = ["manifest.json", "corpus.jsonl", "corpus_embeddings.npy", query_file.name, f"{split}_queries_embeddings.npy"]
    manifest = {"phase": 2, "split": split, "tau": tau, "threshold_status": "provisional correctness check; not calibrated",
                "corpus_size": len(corpus), "query_count": len(queries), "counts": counts,
                "score_definition": "float64 normalized cosine, clipped to [-1,1], inclusive >= tau",
                "ground_truth_independent_of_exact_match": True,
                "repeated_query_check": "passed", "persistence_check": "passed", "phase1_verification": verification,
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "inputs": {name: checksum(source / name) for name in inputs},
                "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": checksum(p)} for p in (result_path, scores_path)}}
    save_json(manifest_path, manifest)
    LOG.info("Phase 2 completed: %s", json.dumps({"split": split, "tau": tau, "corpus_size": len(corpus), "counts": counts, "output": str(destination)}, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--tau", required=True, type=float, help="Explicit threshold; calibration belongs to Phase 4")
    parser.add_argument("--output", default="data/ground_truth")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(output / "phase2.log", encoding="utf-8")])
    try:
        generate(args.config, args.split, args.tau, output)
    except Exception:
        LOG.exception("Phase 2 failed")
        raise


if __name__ == "__main__":
    main()
