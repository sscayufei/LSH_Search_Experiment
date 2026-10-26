"""Offline integration checks against actual downloaded data and model artifacts."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.prepare_data import build_splits, verify_saved

OUTPUT = Path("data/processed")
pytestmark = pytest.mark.skipif(not (OUTPUT / "manifest.json").exists(), reason="Real Phase 1 artifacts have not been generated")


def test_real_source_provenance_and_split_reproduction():
    from datasets import Dataset

    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    arrow_files = list((Path(config["cache"]) / "datasets").rglob(f"{manifest['dataset_revision']}/*.arrow"))
    assert len(arrow_files) == 1
    rows = Dataset.from_file(str(arrow_files[0]))
    rebuilt, _ = build_splits(rows, config)
    for name in ("corpus", "validation_queries", "test_queries"):
        saved = [json.loads(line) for line in (OUTPUT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rebuilt[name] == saved
        if name == "corpus":
            continue
        for query in saved:
            if query["kind"] == "semantic":
                for source in query["sources"]:
                    original = rows[source["pair_id"]]
                    assert original["positive"] == query["text"]
                    assert original["anchor"] == rebuilt["corpus"][source["corpus_id"]]["text"]
    assert verify_saved(OUTPUT)["status"] == "passed"


def test_real_saved_vectors_match_reencoded_text():
    import torch
    from sentence_transformers import SentenceTransformer

    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    torch.set_num_threads(4)
    model = SentenceTransformer(config["embedding"]["model"], revision=manifest["model_revision"],
                                cache_folder=str(Path(config["cache"]) / "models"), device="cpu", local_files_only=True)
    for name in ("corpus", "validation_queries", "test_queries"):
        records = [json.loads(line) for line in (OUTPUT / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        indices = [0, len(records) - 1]
        actual = model.encode([records[i]["text"] for i in indices], normalize_embeddings=True, convert_to_numpy=True)
        saved = np.load(OUTPUT / f"{name}_embeddings.npy", allow_pickle=False)[indices]
        np.testing.assert_allclose(actual, saved, atol=2e-6, rtol=1e-4)
