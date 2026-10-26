"""Offline correctness checks on persisted Phase 2 results; no parameter tuning."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.exact_hash import ExactHashIndex
from src.ground_truth import BruteForceSearch, load_jsonl
from src.prepare_data import checksum

OUTPUT = Path("data/ground_truth")
MANIFEST = OUTPUT / "validation_tau_0.75_manifest.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="Real Phase 2 validation Ground Truth not generated")
def test_real_persisted_ground_truth_against_independent_reference():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = Path("data/processed")
    for filename, digest in manifest["inputs"].items():
        assert checksum(source / filename) == digest
    for filename, info in manifest["outputs"].items():
        assert checksum(OUTPUT / filename) == info["sha256"]
    records = load_jsonl(OUTPUT / "validation_tau_0.75.jsonl")
    queries = load_jsonl(source / "validation_queries.jsonl")
    corpus = load_jsonl(source / "corpus.jsonl")
    x = np.load(source / "corpus_embeddings.npy").astype(np.float64)
    q = np.load(source / "validation_queries_embeddings.npy").astype(np.float64)
    # Independent formula directly on unmodified saved vectors, no oracle helper.
    reference = (q @ x.T) / (np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(x, axis=1)[None, :])
    saved = np.load(OUTPUT / "validation_tau_0.75_scores.npy")
    np.testing.assert_allclose(saved, np.clip(reference, -1, 1), atol=2e-14, rtol=1e-12)
    exact = ExactHashIndex(r["text"] for r in corpus)
    brute = BruteForceSearch(x)
    assert len(records) == 60
    for i, (record, query) in enumerate(zip(records, queries, strict=True)):
        assert record["query_id"] == query["query_id"]
        assert record["kind"] == query["kind"]
        expected = np.flatnonzero(reference[i] >= manifest["tau"]).tolist()
        assert record["ground_truth_ids"] == expected
        assert record["ground_truth_ids"] == brute.search(q[i], manifest["tau"]).tolist()
        expected_exact = [j for j, r in enumerate(corpus) if r["text"] == query["text"]]
        assert record["exact_ids"] == expected_exact == exact.search(query["text"])
