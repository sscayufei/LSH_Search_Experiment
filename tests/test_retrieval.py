import numpy as np
import pytest

from src.random_hyperplane_lsh import RandomHyperplaneLSH
from src.retrieval import HybridRetriever


def test_exact_first_skips_lsh_and_vector_validation(monkeypatch):
    index = RandomHyperplaneLSH(k=2, L=2).fit([[1, 0], [1, 0], [0, 1]])
    retriever = HybridRetriever(["same", "same", "other"], index)

    def forbid(*args, **kwargs):
        raise AssertionError("Exact hit must skip LSH and embedding access")

    monkeypatch.setattr(index, "search_with_stats", forbid)
    result = retriever.search("same", None, 0.75)
    assert result.ids.tolist() == [0, 1]
    assert result.route == "exact"
    assert result.scores is None
    assert result.stats["comparisons"] == result.stats["table_lookups"] == 0


def test_semantic_route_matches_lsh_and_missing_results_stay_empty():
    index = RandomHyperplaneLSH(k=1, L=1).fit([[1.]])
    retriever = HybridRetriever(["original"], index)
    result = retriever.search("different text", np.array([1.]), 0.75)
    assert result.route == "lsh"
    assert result.ids.tolist() == index.search([1.], 0.75).tolist() == [0]
    assert retriever.search("missing", [-1.], -1).ids.tolist() == []


def test_alignment_rejected():
    with pytest.raises(ValueError):
        HybridRetriever([], RandomHyperplaneLSH().fit([[1.]]))
