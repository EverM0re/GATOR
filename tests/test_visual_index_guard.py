"""Visual retrieval must fail loudly, not silently.

campaign wave3 ran with visual recall disabled on every query: the index and the
query encoder disagreed on dimension, numpy raised an opaque gufunc error, and
the retriever swallowed it. The run completed with plausible-looking numbers
produced from half the intended candidate pool.
"""

import numpy as np
import pytest

from file_router.storage.visual_index import VisualIndex


def _index(tmp_path, dim, count=4):
    index = VisualIndex(str(tmp_path), dim)
    for i in range(count):
        index.add(f"n{i}", np.random.default_rng(i).standard_normal(dim).astype(np.float32))
    return index


def test_dimension_mismatch_raises_a_diagnosable_error(tmp_path):
    index = _index(tmp_path, 512)
    with pytest.raises(ValueError) as excinfo:
        index.search(np.zeros(27, dtype=np.float32), topk=3)
    message = str(excinfo.value)
    # The message has to name both dimensions and the remedy, or the next
    # person sees the same opaque failure.
    assert "512" in message and "27" in message
    assert "ingest" in message.lower()


def test_matching_dimension_still_searches(tmp_path):
    index = _index(tmp_path, 512)
    hits = index.search(np.random.default_rng(0).standard_normal(512).astype(np.float32),
                        topk=2)
    assert len(hits) == 2
    assert all(isinstance(h[0], str) for h in hits)


def test_query_shape_is_normalized_before_comparison(tmp_path):
    """A (1, D) query is a valid encoder output and must not be rejected."""
    index = _index(tmp_path, 512)
    query = np.random.default_rng(0).standard_normal((1, 512)).astype(np.float32)
    assert len(index.search(query, topk=2)) == 2


def test_empty_index_returns_nothing_rather_than_raising(tmp_path):
    assert VisualIndex(str(tmp_path), 512).search(np.zeros(27, dtype=np.float32)) == []


def test_dimension_change_during_ingest_is_rejected(tmp_path):
    """A store half-written by one encoder and half by another is unusable.

    Catching it at write time points at the actual cause; catching it at query
    time in a later run produces an opaque matmul error instead.
    """
    index = VisualIndex(str(tmp_path), 512)
    index.add("a", np.ones(512, dtype=np.float32))
    with pytest.raises(ValueError) as excinfo:
        index.add("b", np.ones(2048, dtype=np.float32))
    assert "2048" in str(excinfo.value) and "512" in str(excinfo.value)


def test_first_vector_sets_the_dimension(tmp_path):
    """An empty index accepts whatever the encoder produces."""
    index = VisualIndex(str(tmp_path), 512)
    index.add("a", np.ones(2048, dtype=np.float32))
    assert index.size() == 1
