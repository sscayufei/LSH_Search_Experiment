"""Original-string exact matching; Python dict verifies equality after hashing."""
from collections.abc import Iterable


class ExactHashIndex:
    """Map original text to every matching corpus row ID, in insertion order."""

    def __init__(self, texts: Iterable[str]):
        self._texts = tuple(texts)
        self._buckets: dict[str, list[int]] = {}
        for doc_id, text in enumerate(self._texts):
            if not isinstance(text, str):
                raise TypeError("Corpus text must be a string")
            self._buckets.setdefault(text, []).append(doc_id)

    def search(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Query text must be a string")
        # dict compares full strings on hash collisions; retain an explicit raw-text check.
        return [doc_id for doc_id in self._buckets.get(text, ()) if self._texts[doc_id] == text]
