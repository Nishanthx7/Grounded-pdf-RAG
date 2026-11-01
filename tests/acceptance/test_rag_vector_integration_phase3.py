"""Offline cross-module acceptance test: real vector manager, fake providers."""
from pdf_rag.application.config import GenerationPolicy, RetrievalPolicy
from pdf_rag.application.services import DefaultRAGPipeline
from pdf_rag.domain.models import AnswerStatus
from pdf_rag.infrastructure.vector.manager import SessionVectorStoreManager
from tests._builders import (
    FakeEmbeddingProvider, FakeGenerator, FakeVectorFactory, chunk, config,
)


def test_vector_retrieval_to_cited_answer_uses_real_application_services():
    c1 = chunk(text="The result is 42.", page=7)
    c2 = chunk(text="The weather is hot.", page=2)
    embeddings = FakeEmbeddingProvider({
        c1.text: (1.0, 0.0), c2.text: (0.0, 1.0), "What result?": (1.0, 0.0),
    })
    manager = SessionVectorStoreManager(embeddings, FakeVectorFactory())
    idx = manager.build_index((c1, c2), config=config())
    gen = FakeGenerator()
    response = DefaultRAGPipeline(manager, gen).answer(
        idx, "What result?", retrieval=RetrievalPolicy(top_k=1), generation=GenerationPolicy(),
    )
    assert response.status == AnswerStatus.ANSWERED
    assert len(response.citations) == 1
    assert response.citations[0].page_number == 7
    assert response.citations[0].chunk_id == c1.chunk_id
    assert embeddings.document_calls == [(c1.text, c2.text)]
    assert embeddings.query_calls == ["What result?"]


def test_real_manager_with_no_relevant_hits_never_calls_generator():
    c = chunk(text="Evidence about geography.")
    embeddings = FakeEmbeddingProvider({c.text: (1., 0.), "unrelated": (-1., 0.)})
    manager = SessionVectorStoreManager(embeddings, FakeVectorFactory())
    idx = manager.build_index((c,), config=config())
    gen = FakeGenerator()
    response = DefaultRAGPipeline(manager, gen).answer(
        idx, "unrelated", retrieval=RetrievalPolicy(similarity_threshold=.7),
        generation=GenerationPolicy(),
    )
    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert gen.calls == []
