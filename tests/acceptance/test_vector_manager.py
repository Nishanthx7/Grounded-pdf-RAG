"""RED: index lifecycle, normalized vector search, and retrieval contracts."""
import math
import pytest
from pdf_rag.application.services import SessionVectorStoreManager
from pdf_rag.application.config import RetrievalPolicy
from pdf_rag.domain.errors import EmbeddingError, IndexBuildError, IndexNotFoundError, RetrievalError
from tests._builders import FakeEmbeddingProvider, FakeVectorFactory, chunk, config, index

pytestmark = pytest.mark.implementation


def make_manager(vectors, *, dim=768):
    embedding = FakeEmbeddingProvider(vectors, dimension=dim)
    factory = FakeVectorFactory()
    return SessionVectorStoreManager(embedding, factory), embedding, factory


def test_index_builds_once_with_exactly_one_vector_per_chunk():
    cs = (chunk(text="alpha"), chunk(text="beta"))
    manager, embedding, factory = make_manager({"alpha": (3.0, 0.0), "beta": (0.0, 5.0)})
    descriptor = manager.build_index(cs, config=config())
    assert descriptor.chunk_count == 2
    assert descriptor.document_count == 1
    assert descriptor.embedding_dimension == 768
    assert embedding.document_calls == [("alpha", "beta")]
    assert len(factory.built_vectors) == 1
    assert len(factory.built_vectors[0]) == 2
    for vector in factory.built_vectors[0]:
        assert math.isclose(sum(x*x for x in vector), 1.0, abs_tol=1e-6)


def test_retrieval_normalizes_query_and_filters_low_scores():
    cs = tuple(chunk(text=x) for x in ["alpha", "diagonal", "opposite"])
    values = {"alpha": (3, 0), "diagonal": (1, 1), "opposite": (-1, 0), "q": (9, 0)}
    manager, embedding, factory = make_manager(values)
    descriptor = manager.build_index(cs, config=config())
    result = manager.retrieve(descriptor, "q", policy=RetrievalPolicy(top_k=3, similarity_threshold=.70))
    assert [h.chunk.text for h in result.hits] == ["alpha", "diagonal"]
    assert result.hits[0].similarity_score == pytest.approx(1.0, abs=1e-5)
    assert result.hits[1].similarity_score == pytest.approx(2**(-.5), abs=1e-5)
    assert embedding.query_calls == ["q"]
    norm = sum(v*v for v in factory.backends[0].query_calls[0][0])
    assert norm == pytest.approx(1.0)


def test_none_above_similarity_threshold_returns_valid_empty_result():
    manager, _, _ = make_manager({"alpha": (1, 0), "q": (-1, 0)})
    descriptor = manager.build_index((chunk(text="alpha"),), config=config())
    result = manager.retrieve(descriptor, "q", policy=RetrievalPolicy(top_k=5, similarity_threshold=.7))
    assert result.hits == ()
    assert result.index_id == descriptor.index_id


def test_equal_similarity_scores_are_tiebroken_by_chunk_id():
    cs = (chunk(text="alpha"), chunk(text="beta"))
    manager, _, _ = make_manager({"alpha": (1, 0), "beta": (1, 0), "q": (1, 0)})
    idx = manager.build_index(cs, config=config())
    result = manager.retrieve(idx, "q", policy=RetrievalPolicy(top_k=2, similarity_threshold=.7))
    assert [h.chunk.chunk_id for h in result.hits] == sorted(c.chunk_id for c in cs)
    assert [h.rank for h in result.hits] == [1, 2]


@pytest.mark.parametrize("raw_vectors", [
    {"alpha": (0, 0)},
    {"alpha": (float("nan"), 1)},
    {"alpha": (float("inf"), 1)},
])
def test_zero_nan_inf_vectors_are_rejected_before_faiss(raw_vectors):
    manager, _, factory = make_manager(raw_vectors)
    with pytest.raises(EmbeddingError):
        manager.build_index((chunk(text="alpha"),), config=config())
    assert factory.built_vectors == []


def test_embedding_dimension_mismatch_rejected():
    manager, _, factory = make_manager({"alpha": (1, 0)}, dim=4)
    with pytest.raises(EmbeddingError):
        manager.build_index((chunk(text="alpha"),), config=config())
    assert factory.built_vectors == []


def test_duplicate_chunk_ids_rejected():
    c = chunk(text="alpha")
    manager, _, _ = make_manager({"alpha": (1, 0)})
    with pytest.raises(IndexBuildError):
        manager.build_index((c, c), config=config())


def test_empty_index_build_rejected():
    manager, _, _ = make_manager({})
    with pytest.raises(IndexBuildError):
        manager.build_index((), config=config())


def test_clear_invalidates_index_handle():
    manager, _, _ = make_manager({"alpha": (1, 0), "q": (1, 0)})
    descriptor = manager.build_index((chunk(text="alpha"),), config=config())
    manager.clear()
    with pytest.raises(IndexNotFoundError):
        manager.retrieve(descriptor, "q", policy=RetrievalPolicy())


def test_foreign_index_descriptor_rejected():
    manager, _, _ = make_manager({"alpha": (1, 0), "q": (1, 0)})
    foreign = index((chunk(text="alpha"),))
    with pytest.raises(IndexNotFoundError):
        manager.retrieve(foreign, "q", policy=RetrievalPolicy())


def test_failed_replacement_preserves_previous_ready_index():
    manager, embedding, _ = make_manager({"alpha": (1, 0), "q": (1, 0)})
    previous = manager.build_index((chunk(text="alpha"),), config=config())
    embedding.error = EmbeddingError("temporary embedding failure")
    with pytest.raises(EmbeddingError):
        manager.build_index((chunk(text="other"),), config=config())
    embedding.error = None
    assert manager.retrieve(previous, "q", policy=RetrievalPolicy()).hits


def test_blank_query_rejected_before_embedding_call():
    manager, embedding, _ = make_manager({"alpha": (1, 0)})
    descriptor = manager.build_index((chunk(text="alpha"),), config=config())
    with pytest.raises(RetrievalError):
        manager.retrieve(descriptor, "  ", policy=RetrievalPolicy())
    assert embedding.query_calls == []
