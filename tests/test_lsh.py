import math

import numpy as np
import pytest

import src.random_hyperplane_lsh as lsh_module
from src.ground_truth import BruteForceSearch
from src.random_hyperplane_lsh import RandomHyperplaneLSH, candidate_probability, pack_signatures


def test_little_endian_signatures_support_more_than_64_bits():
    assert pack_signatures(np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)) == [5, 2]
    bits = np.zeros((1, 70), dtype=bool)
    bits[0, [0, 64, 69]] = True
    assert pack_signatures(bits) == [1 + 2**64 + 2**69]


def test_table_membership_signatures_zero_boundary_and_immutability():
    x = np.array([[1., 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    index = RandomHyperplaneLSH(k=3, L=4).fit(x)
    planes = index.hyperplanes
    for table_id, table in enumerate(index.tables):
        all_ids = [i for ids in table.values() for i in ids]
        assert sorted(all_ids) == list(range(len(x)))
        for i, vector in enumerate(x):
            expected = sum((1 << j) for j, r in enumerate(planes[table_id]) if np.dot(vector, r) >= 0)
            assert index.signatures(vector)[table_id] == expected
            assert i in table[expected]
    assert index.signatures(x[0]) == index.signatures(x[4])
    assert index.fit_stats["id_references"] == len(x) * index.L
    with pytest.raises(TypeError):
        index.tables[0][999] = (0,)
    planes[:] = 0
    assert np.any(index.hyperplanes != 0)


def test_exact_zero_projection_sets_one(monkeypatch):
    class AxisRNG:
        def normal(self, size):
            assert size == (1, 2, 2)
            return np.array([[[1., 0], [0, 1]]])

    monkeypatch.setattr(np.random, "default_rng", lambda seed: AxisRNG())
    index = RandomHyperplaneLSH(k=2, L=1).fit([[1, 0], [0, 1], [-1, 0], [0, -1]])
    assert [index.signatures(q)[0] for q in ([1, 0], [0, 1], [-1, 0], [0, -1])] == [3, 3, 2, 1]


def test_seed_reproduction_and_no_query_time_randomness(monkeypatch):
    x = np.random.default_rng(2).normal(size=(30, 5))
    first = RandomHyperplaneLSH(7, 3, 42).fit(x)
    second = RandomHyperplaneLSH(7, 3, 42).fit(x)
    third = RandomHyperplaneLSH(7, 3, 43).fit(x)
    np.testing.assert_array_equal(first.hyperplanes, second.hyperplanes)
    assert first.tables == second.tables
    assert not np.array_equal(first.hyperplanes, third.hyperplanes)

    def forbid_rng(*args, **kwargs):
        raise AssertionError("Queries must reuse saved hyperplanes")

    monkeypatch.setattr(np.random, "default_rng", forbid_rng)
    for q in x[:5]:
        assert first.signatures(q) == second.signatures(q)
        np.testing.assert_array_equal(first.candidates(q), second.candidates(q))
        np.testing.assert_array_equal(first.search(q, 0.5), first.search(q, 0.5))


def test_candidate_union_deduplicated_and_only_candidates_are_scored(monkeypatch):
    x = np.random.default_rng(5).normal(size=(50, 6))
    query = x[0]
    index = RandomHyperplaneLSH(k=4, L=5).fit(x)
    ids = index.candidates(query)
    expected = set()
    for signature, table in zip(index.signatures(query), index.tables):
        expected.update(table.get(signature, ()))
    assert ids.tolist() == sorted(expected)
    assert len(ids) == len(set(ids.tolist()))
    assert 0 in ids
    assert set(ids) <= set(range(len(x)))
    assert len(ids) < len(x)
    actual_scored = []
    original = index._score_candidates
    original_cosine = lsh_module.cosine_scores_normalized
    scored_row_counts = []

    def cosine_spy(vectors, normalized_query):
        scored_row_counts.append(len(vectors))
        return original_cosine(vectors, normalized_query)

    def spy(normalized_query, candidate_ids):
        actual_scored.extend(candidate_ids.tolist())
        return original(normalized_query, candidate_ids)

    monkeypatch.setattr(index, "_score_candidates", spy)
    monkeypatch.setattr(lsh_module, "cosine_scores_normalized", cosine_spy)
    result = index.search_with_stats(query, 0.3)
    assert actual_scored == ids.tolist()
    assert scored_row_counts == [len(ids)]
    assert result.stats["comparisons"] == len(ids)
    assert result.stats["table_lookups"] == 5
    truth = BruteForceSearch(x).search(query, 0.3)
    assert result.ids.tolist() == sorted(set(ids) & set(truth))


def test_empty_bucket_does_not_fallback_or_score(monkeypatch):
    index = RandomHyperplaneLSH(k=1, L=1).fit([[1.]])

    def forbid(*args, **kwargs):
        raise AssertionError("Empty candidates must not trigger vector comparisons")

    monkeypatch.setattr(index, "_score_candidates", forbid)
    monkeypatch.setattr(BruteForceSearch, "search", forbid)
    result = index.search_with_stats([-1.], -1)
    assert result.candidate_ids.tolist() == result.ids.tolist() == []
    assert result.stats["comparisons"] == 0


def test_full_candidate_scan_matches_oracle_at_exact_score_thresholds():
    # All positive rows collide for d=1 in every table. Tau=1 includes all rows.
    index = RandomHyperplaneLSH(k=2, L=3).fit([[2.], [4.], [-1.]])
    assert index.search([3.], 1).tolist() == [0, 1]
    # Arbitrary-dimensional threshold copied exactly from oracle scores.
    x = np.random.default_rng(19).normal(size=(30, 11))
    query = x[0]
    index = RandomHyperplaneLSH(k=2, L=3).fit(x)
    brute = BruteForceSearch(x)
    candidates = index.candidates(query)
    for tau in brute.scores(query)[candidates]:
        assert index.search(query, tau).tolist() == sorted(set(candidates) & set(brute.search(query, tau)))


def test_empty_index_bucket_stats_and_refit():
    index = RandomHyperplaneLSH(k=2, L=3).fit(np.empty((0, 2)))
    assert index.search([1, 0], 0).tolist() == []
    assert all(s["empty_bucket_ratio"] == 1 and s["max_bucket_size"] == 0 for s in index.bucket_stats())
    index.fit([[1, 0], [1, 0]])
    assert index.search([1, 0], 1).tolist() == [0, 1]
    assert all(s["id_references"] == 2 and s["max_bucket_size"] == 2 for s in index.bucket_stats())


def test_input_copy_and_wide_k():
    x = np.array([[1., 0], [0, 1]])
    index = RandomHyperplaneLSH(k=70, L=2).fit(x)
    signatures = index.signatures(x[0])
    x[:] = 0
    assert index.signatures([1, 0]) == signatures
    assert index.search([1, 0], 1).tolist() == [0]


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"L": -1}, {"k": 1.5}, {"L": True}, {"random_state": None}, {"random_state": -1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RandomHyperplaneLSH(**kwargs)


def test_unfitted_and_invalid_data():
    with pytest.raises(RuntimeError):
        RandomHyperplaneLSH().candidates([1, 0])
    with pytest.raises(ValueError):
        RandomHyperplaneLSH().fit([[0, 0]])
    index = RandomHyperplaneLSH().fit([[1, 0]])
    for query in ([0, 0], [np.nan, 0], [1, 2, 3]):
        with pytest.raises(ValueError):
            index.candidates(query)
    with pytest.raises(ValueError):
        index.search([1, 0], 1.5)


def test_theoretical_probability_endpoints_and_monotonicity():
    assert candidate_probability(1, 12, 16) == 1
    assert candidate_probability(-1, 12, 16) == 0
    assert candidate_probability(0, 1, 1) == pytest.approx(0.5)
    assert candidate_probability(0.75, 12, 16) == pytest.approx(1 - (1 - (1 - math.acos(0.75) / math.pi)**12)**16)
    assert candidate_probability(0.75, 8, 16) > candidate_probability(0.75, 12, 16)
    assert candidate_probability(0.75, 12, 32) > candidate_probability(0.75, 12, 16)
    for value in (1.1, -1.1, np.nan, "0.75"):
        with pytest.raises(ValueError):
            candidate_probability(value, 12, 16)
