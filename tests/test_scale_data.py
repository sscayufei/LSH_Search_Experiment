import numpy as np
import pytest

from src.prepare_scale_data import build_nested_records, validate_vectors


def test_nested_prefix_query_exclusion_and_reproducible_extension_provenance():
    base = [{"doc_id": 0, "text": "base"}]
    rows = [{"anchor": "base", "positive": "p1"}, {"anchor": "a2", "positive": "query"},
            {"anchor": "p1", "positive": "p2"}, {"anchor": "", "positive": "p3"}]
    small, available = build_nested_records(base, rows, {"query", "base"}, 3, 42)
    large, _ = build_nested_records(base, rows, {"query", "base"}, 5, 42)
    repeated, _ = build_nested_records(base, rows, {"query", "base"}, 5, 42)
    assert available == 5 and large[:3] == small and large == repeated
    assert large[:1] == base
    assert "query" not in {r["text"] for r in large}
    for record in large[1:]:
        assert rows[record["source_pair_id"]][record["source_column"]] == record["text"]
    assert [r["doc_id"] for r in large] == list(range(5))
    assert len({r["text"] for r in large}) == 5
    with pytest.raises(ValueError, match="only 5"):
        build_nested_records(base, rows, {"query"}, 6, 42)


def test_scale_vector_validation_rejects_substitutes_and_wrong_shapes():
    vectors = np.array([[1., 0], [0, 1]], dtype=np.float32)
    assert validate_vectors(vectors, 2, 2) == 0
    for bad in (np.zeros((2, 2), dtype=np.float32), vectors.astype(np.float64), vectors[:, :1], vectors * np.nan):
        with pytest.raises(ValueError):
            validate_vectors(bad, 2, 2)
