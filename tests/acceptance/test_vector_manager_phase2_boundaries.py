"""Extra Phase 2 RED->GREEN tests. No Google calls and no FAISS requirement."""
import pytest

from pdf_rag.application.services import SessionVectorStoreManager
from pdf_rag.application.config import RetrievalPolicy
from pdf_rag.domain.errors import EmbeddingError, IndexBuildError, IndexNotFoundError, RetrievalError
from pdf_rag.application.ports import VectorMatch
from tests._builders import chunk, config, FakeEmbeddingProvider, FakeVectorFactory, digest

pytestmark = pytest.mark.implementation


def setup_manager():
    embedding = FakeEmbeddingProvider({"alpha": (1, 0), "beta": (0, 1), "q": (1, 0)})
    factory = FakeVectorFactory()
    return SessionVectorStoreManager(embedding, factory), embedding, factory


def test_rebuilding_identical_corpus_reuses_index_and_embeddings():
    manager, provider, factory = setup_manager()
    cs = (chunk(text="alpha"),)
    a = manager.build_index(cs, config=config())
    b = manager.build_index(cs, config=config())
    assert a is b
    assert provider.document_calls == [("alpha",)]
    assert len(factory.built_vectors) == 1


def test_change_embedding_profile_forces_reindexing_and_invalidates_old_descriptor():
    manager, provider, _ = setup_manager()
    cs = (chunk(text="alpha"),)
    first = manager.build_index(cs, config=config())
    alt = config().model_copy(update={"embedding_profile_id": digest("new-profile")})
    second = manager.build_index(cs, config=alt)
    assert first.index_id != second.index_id
    assert len(provider.document_calls) == 2
    with pytest.raises(IndexNotFoundError):
        manager.retrieve(first, "q", policy=RetrievalPolicy())
    assert manager.retrieve(second, "q", policy=RetrievalPolicy()).hits


def test_retrieval_policy_changes_do_not_rebuild_vectors():
    manager, provider, _ = setup_manager()
    idx = manager.build_index((chunk(text="alpha"), chunk(text="beta")), config=config())
    assert len(manager.retrieve(idx, "q", policy=RetrievalPolicy(top_k=1)).hits) == 1
    assert len(manager.retrieve(idx, "q", policy=RetrievalPolicy(top_k=2, similarity_threshold=-1)).hits) == 2
    assert len(provider.document_calls) == 1
    assert len(provider.query_calls) == 2


def test_wrong_embedding_response_count_rejected_atomically():
    class ShortEmbedding(FakeEmbeddingProvider):
        def embed_documents(self, texts):
            return [[1.0] + [0.] * 767]
    manager = SessionVectorStoreManager(ShortEmbedding({}), FakeVectorFactory())
    with pytest.raises(EmbeddingError, match="count"):
        manager.build_index((chunk(text="alpha"), chunk(text="beta")), config=config())


def test_unexpected_embedder_exception_translates_without_leaking_secrets():
    manager, provider, _ = setup_manager()
    provider.error = RuntimeError("google secret test-key-123")
    with pytest.raises(EmbeddingError) as exc:
        manager.build_index((chunk(text="alpha"),), config=config())
    assert "test-key" not in str(exc.value)


def test_backend_build_failure_preserves_existing_index():
    manager, provider, factory = setup_manager()
    descriptor = manager.build_index((chunk(text="alpha"),), config=config())
    def fail(_vectors):
        raise RuntimeError("faiss failed")
    factory.build = fail
    with pytest.raises(IndexBuildError):
        manager.build_index((chunk(text="beta"),), config=config())
    assert manager.retrieve(descriptor, "q", policy=RetrievalPolicy()).hits


def test_backend_invalid_positions_and_scores_raise_retrieval_errors():
    manager, _, factory = setup_manager()
    descriptor = manager.build_index((chunk(text="alpha"),), config=config())
    backend = factory.backends[0]
    original = backend.search
    for invalid in ([VectorMatch(-1, 1.0)], [VectorMatch(3, 0.9)],
                    [VectorMatch(0, float("nan"))], [VectorMatch(0, 3.0)],
                    [VectorMatch(0, 1.0), VectorMatch(0, 1.0)]):
        backend.search = lambda *args, results=invalid, **kwargs: results
        with pytest.raises(RetrievalError):
            manager.retrieve(descriptor, "q", policy=RetrievalPolicy())
    backend.search = original


def test_top_k_tie_break_is_global_even_when_ties_cross_k_boundary():
    cs = (chunk(text="beta"), chunk(text="alpha"))
    manager, provider, factory = setup_manager()
    provider.vectors["beta"] = (1, 0)
    idx = manager.build_index(cs, config=config())
    hit = manager.retrieve(idx, "q", policy=RetrievalPolicy(top_k=1)).hits[0]
    assert hit.chunk.chunk_id == min(c.chunk_id for c in cs)


def test_provider_profile_mismatch_is_rejected_before_embedding():
    manager, provider, _ = setup_manager()
    provider.embedding_profile_id = digest("incorrect")
    with pytest.raises(IndexBuildError, match="profile"):
        manager.build_index((chunk(text="alpha"),), config=config())
    assert provider.document_calls == []


def test_changed_chunk_text_cannot_reuse_fingerprint_even_if_id_was_reused():
    manager, provider, _ = setup_manager()
    original = chunk(text="alpha")
    old_idx = manager.build_index((original,), config=config())
    altered = original.model_copy(update={"text": "beta"})
    new_idx = manager.build_index((altered,), config=config())
    assert old_idx.index_id != new_idx.index_id
    assert provider.document_calls == [("alpha",), ("beta",)]
