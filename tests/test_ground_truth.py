import numpy as np
import pytest

from src.ground_truth import BruteForceSearch, normalize_rows, threshold_ids


def test_identical_orthogonal_opposite_and_inclusive_boundaries():
    index = BruteForceSearch([[2, 0], [0, 4], [-3, 0], [2, 2]])
    query = np.array([7, 0])
    np.testing.assert_allclose(index.scores(query), [1, 0, -1, 1 / np.sqrt(2)], atol=1e-15)
    assert index.search(query, 1).tolist() == [0]
    assert index.search(query, 0).tolist() == [0, 1, 3]
    assert index.search(query, -1).tolist() == [0, 1, 2, 3]
    assert threshold_ids(np.array([0.75, np.nextafter(0.75, -np.inf), np.nextafter(0.75, np.inf)]), 0.75).tolist() == [0, 2]


def test_against_independent_scalar_cosine_reference():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(37, 9))
    query = rng.normal(size=9)
    expected = np.array([sum(float(a) * float(b) for a, b in zip(row, query)) /
                         (np.sqrt(sum(float(a) ** 2 for a in row)) * np.sqrt(sum(float(b) ** 2 for b in query))) for row in vectors])
    index = BruteForceSearch(vectors)
    np.testing.assert_allclose(index.scores(query), expected, atol=1e-15)
    expected_ids = [i for i, score in enumerate(expected) if score >= 0.1]
    assert index.search(query, 0.1).tolist() == expected_ids
    assert index.search(query, 0.1).tolist() == expected_ids


def test_source_array_mutation_does_not_change_oracle():
    vectors = np.array([[1.0, 0], [0, 1]])
    index = BruteForceSearch(vectors)
    vectors[:] = -1
    assert index.search([1, 0], 0.5).tolist() == [0]


def test_empty_corpus_and_empty_relevant_set():
    empty = BruteForceSearch(np.empty((0, 2)))
    assert empty.scores([1, 0]).shape == (0,)
    assert empty.search([1, 0], 0.5).tolist() == []
    assert BruteForceSearch([[0, 1]]).search([1, 0], 0.5).tolist() == []


def test_extreme_nonzero_magnitudes_normalize_safely():
    vectors = normalize_rows([[1e300, 1e300], [1e-300, 1e-300]])
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
    np.testing.assert_allclose(vectors[0], vectors[1])


@pytest.mark.parametrize("vectors", [[[0, 0]], [[np.nan, 0]], [[np.inf, 1]], [1, 2], np.empty((2, 0)), [[1j, 0]], [["a", "b"]]])
def test_invalid_corpus_rejected(vectors):
    with pytest.raises(ValueError):
        BruteForceSearch(vectors)


@pytest.mark.parametrize("query", [[0, 0], [np.nan, 1], [np.inf, 1], [1, 2, 3], [[1, 2]], [1j, 0]])
def test_invalid_query_rejected(query):
    with pytest.raises(ValueError):
        BruteForceSearch([[1, 0]]).scores(query)


@pytest.mark.parametrize("tau", [np.nan, np.inf, -1.01, 1.01, [0.5], 1j, "0.5"])
def test_invalid_threshold_rejected(tau):
    with pytest.raises(ValueError):
        BruteForceSearch([[1, 0]]).search([1, 0], tau)
