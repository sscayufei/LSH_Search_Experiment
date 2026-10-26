import pytest

from src.scale_analysis import prediction_rows


def test_theory_extrapolation_and_storage_are_separate_and_not_measurements():
    baseline = {"N": 10000, "mean_candidates": 150.0, "mean_false_candidates": 145.0}
    rows = prediction_rows(baseline, [10000, 100000, 1000000], [0.0], 12, 32, 384)
    assert not any(row["value_type"] == "measured" for row in rows)
    candidates = [r for r in rows if r["quantity"] == "mean_candidates"]
    assert [r["value"] for r in candidates] == [150, 1500, 15000]
    assert all(r["value_type"] == "extrapolated_from_measured_10k" for r in candidates)
    theory = [r for r in rows if r["value_type"] == "conditional_theory"]
    assert theory[1]["value"] == pytest.approx(theory[0]["value"] * 10)
    assert theory[2]["value"] == pytest.approx(theory[0]["value"] * 100)
    storage = [r for r in rows if r["quantity"] == "float32_vector_payload"]
    assert storage[-1]["value"] == 1536000000
    assert all(r["units"] == "bytes" and r["value_type"] == "arithmetic_storage_formula" for r in storage)
    assert not any(r["quantity"] in ("query_time_ms", "index_bytes") for r in rows)


def test_invalid_prediction_sizes_and_probability_rejected():
    with pytest.raises(ValueError):
        prediction_rows({"N": 0}, [10000], [0.0], 12, 32, 384)
    with pytest.raises(ValueError):
        prediction_rows({"N": 10000, "mean_candidates": 1, "mean_false_candidates": 1}, [-1], [0.0], 12, 32, 384)
    with pytest.raises(ValueError):
        prediction_rows({"N": 10000, "mean_candidates": 1, "mean_false_candidates": 1}, [10000], [1.1], 12, 32, 384)
