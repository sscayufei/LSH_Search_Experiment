import json

import numpy as np
import pytest

from src.prepare_data import build_splits, checksum, save_json, validate, verify_saved


@pytest.fixture
def config():
    return {"seed": 42, "dataset": {"corpus_size": 6, "validation_semantic_queries": 2,
            "test_semantic_queries": 2, "validation_exact_queries": 1, "test_exact_queries": 1},
            "embedding": {"dimension": 3}}


@pytest.fixture
def rows():
    return [{"anchor": f"question {i}", "positive": f"related {i}"} for i in range(6)]


def unit_vectors(splits):
    # Synthetic vectors are restricted to unit tests; never used by the pipeline.
    return {k: np.tile(np.array([[1, 0, 0]], dtype=np.float32), (len(v), 1)) for k, v in splits.items()}


def test_reproducible_unique_disjoint(config, rows):
    a, stats = build_splits(rows + rows, config)
    b, _ = build_splits(rows + rows, config)
    assert a == b
    assert stats["anchor_duplicate_ratio"] == 0.5
    assert len(a["corpus"]) == 6
    assert not ({r["text"] for r in a["validation_queries"]} & {r["text"] for r in a["test_queries"]})
    validate(a, unit_vectors(a), config)
    for split in ("validation_queries", "test_queries"):
        for query in a[split]:
            if query["kind"] == "semantic":
                assert len(query["sources"]) == 2
                for source in query["sources"]:
                    assert (rows + rows)[source["pair_id"]]["positive"] == query["text"]
                    assert a["corpus"][source["corpus_id"]]["text"] == (rows + rows)[source["pair_id"]]["anchor"]


def test_original_strings_and_invalid_rows(config, rows):
    rows = rows + [{"anchor": " question 0 ", "positive": "another"},
                   {"anchor": "", "positive": "bad"}, {"anchor": None, "positive": "bad"},
                   {"anchor": "bad", "positive": "  "}]
    config["dataset"]["corpus_size"] = 7
    splits, stats = build_splits(rows, config)
    assert stats["unique_anchors"] == 7
    assert stats["invalid_rows"] == 3
    assert {" question 0 ", "question 0"} <= {r["text"] for r in splits["corpus"]}


def test_query_overlap_removed_and_sources_retained(config, rows):
    rows += [{"anchor": "question 0", "positive": "question 1"},
             {"anchor": "question 1", "positive": "related 0"}]
    splits, stats = build_splits(rows, config)
    assert stats["eligible_unique_semantic_queries"] == 6
    assert all(r["text"] != "question 1" for split in ("validation_queries", "test_queries") for r in splits[split] if r["kind"] == "semantic")


@pytest.mark.parametrize("case", ["corpus", "semantic", "exact"])
def test_insufficient_data_fails(config, rows, case):
    if case == "corpus":
        config["dataset"]["corpus_size"] = 7
    elif case == "semantic":
        config["dataset"]["test_semantic_queries"] = 10
    else:
        config["dataset"]["test_exact_queries"] = 7
    with pytest.raises(ValueError):
        build_splits(rows, config)


def test_empty_input_and_negative_counts(config, rows):
    with pytest.raises(ValueError):
        build_splits([], config)
    config["dataset"]["validation_semantic_queries"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        build_splits(rows, config)


@pytest.mark.parametrize("case", ["nan", "zero", "dimension", "dtype", "overlap", "id", "source", "count"])
def test_validation_rejects_bad_artifacts(config, rows, case):
    splits, _ = build_splits(rows, config)
    vectors = unit_vectors(splits)
    if case == "nan":
        vectors["corpus"][0, 0] = np.nan
    elif case == "zero":
        vectors["corpus"][0] = 0
    elif case == "dimension":
        vectors["corpus"] = vectors["corpus"][:, :2]
    elif case == "dtype":
        vectors["corpus"] = vectors["corpus"].astype(np.float64)
    elif case == "overlap":
        splits["test_queries"][0]["text"] = splits["validation_queries"][0]["text"]
    elif case == "id":
        splits["corpus"][0]["doc_id"] = 8
    elif case == "source":
        splits["test_queries"][0]["sources"][0]["corpus_id"] = 999
    else:
        config["dataset"]["test_semantic_queries"] = 3
    with pytest.raises(ValueError):
        validate(splits, vectors, config)


def test_disk_reload_and_corruption_detection(tmp_path, config, rows):
    splits, _ = build_splits(rows, config)
    vectors = unit_vectors(splits)
    files = {}
    for name, records in splits.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        np.save(tmp_path / f"{name}_embeddings.npy", vectors[name])
    for path in tmp_path.iterdir():
        files[path.name] = {"sha256": checksum(path)}
    save_json(tmp_path / "manifest.json", {"files": files, "config": config})
    assert verify_saved(tmp_path)["status"] == "passed"
    with (tmp_path / "corpus.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Checksum"):
        verify_saved(tmp_path)
