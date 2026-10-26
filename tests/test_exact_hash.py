import pytest

from src.exact_hash import ExactHashIndex


def test_original_strings_and_duplicates():
    index = ExactHashIndex(["Question", "question", " Question", "Question ", "Question", "", "é", "e\u0301"])
    assert index.search("Question") == [0, 4]
    assert index.search("question") == [1]
    assert index.search(" Question") == [2]
    assert index.search("Question ") == [3]
    assert index.search("") == [5]
    assert index.search("é") == [6]
    assert index.search("e\u0301") == [7]
    assert index.search("missing") == []


def test_real_hash_collision_checks_string_equality():
    class CollidingString(str):
        def __hash__(self):
            return 7

    first, second, missing = map(CollidingString, ("first", "second", "missing"))
    assert hash(first) == hash(second) == hash(missing)
    assert first != second
    index = ExactHashIndex([first, second, first])
    assert index.search(CollidingString("first")) == [0, 2]
    assert index.search(CollidingString("second")) == [1]
    assert index.search(missing) == []


def test_empty_index_and_result_mutation():
    assert ExactHashIndex([]).search("any") == []
    texts = ["a"]
    index = ExactHashIndex(texts)
    texts[0] = "changed"
    result = index.search("a")
    result.append(99)
    assert index.search("a") == [0]


def test_invalid_types_rejected():
    with pytest.raises(TypeError):
        ExactHashIndex(["valid", None])
    with pytest.raises(TypeError):
        ExactHashIndex(["valid"]).search(1)
