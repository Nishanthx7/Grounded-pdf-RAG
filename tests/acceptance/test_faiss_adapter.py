"""RED: concrete FAISS adapter must honor Step 3 vector search protocol."""
import pytest
pytest.importorskip("faiss")
from pdf_rag.infrastructure.vector.faiss_backend import FaissVectorSearchBackendFactory
pytestmark = [pytest.mark.implementation, pytest.mark.integration]


def test_factory_search_returns_cosine_scores_and_positions():
    factory = FaissVectorSearchBackendFactory()
    backend = factory.build([[3., 0.], [1., 1.], [-1., 0.]])
    hits = backend.search([9., 0.], top_k=3)
    assert [m.position for m in hits] == [0, 1, 2]
    assert [m.similarity_score for m in hits] == pytest.approx([1., 2**-.5, -1.], abs=1e-5)


def test_factory_supports_small_top_k():
    backend = FaissVectorSearchBackendFactory().build([[1., 0.], [0., 1.]])
    hits = backend.search([0., 4.], top_k=1)
    assert len(hits) == 1
    assert hits[0].position == 1
