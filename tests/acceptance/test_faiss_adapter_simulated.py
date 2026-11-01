"""FAISS boundary simulation that runs without the faiss-cpu wheel.

This is NOT a substitute for tests/acceptance/test_faiss_adapter.py,
which runs the production FAISS C++ extension on the user's machine.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pdf_rag.domain.errors import IndexBuildError, RetrievalError
from pdf_rag.infrastructure.vector.faiss_backend import FaissVectorSearchBackendFactory

pytestmark = pytest.mark.implementation


class InMemoryIP:
    def __init__(self, dimension):
        self.dimension = dimension
        self.data = None

    def add(self, vectors):
        assert vectors.dtype == np.float32
        assert vectors.flags["C_CONTIGUOUS"]
        self.data = vectors.copy()

    def search(self, query, k):
        scores = self.data @ query[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return scores[order].reshape(1, -1), order.astype("int64").reshape(1, -1)


def test_factory_preserves_cosine_contract_with_numpy_faiss_test_double(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(IndexFlatIP=InMemoryIP))
    backend = FaissVectorSearchBackendFactory().build([[3, 0], [1, 1], [-1, 0]])
    results = backend.search([9, 0], top_k=3)
    assert [r.position for r in results] == [0, 1, 2]
    assert [r.similarity_score for r in results] == pytest.approx([1., 2 ** -0.5, -1.], abs=1e-5)


def test_faiss_factory_rejects_bad_vectors_before_native_index(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(IndexFlatIP=InMemoryIP))
    factory = FaissVectorSearchBackendFactory()
    for values in ([], [[0, 0]], [[1, 0], [1]], [[float("nan"), 0]]):
        with pytest.raises(IndexBuildError):
            factory.build(values)


def test_backend_rejects_bad_query_dimension_and_norm(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(IndexFlatIP=InMemoryIP))
    backend = FaissVectorSearchBackendFactory().build([[1, 0], [0, 1]])
    for value in ([], [0, 0], [1, 0, 0], [float("inf"), 0]):
        with pytest.raises(RetrievalError):
            backend.search(value, top_k=2)
