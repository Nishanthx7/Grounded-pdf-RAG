"""Deterministic test-data builders; never call Gemini or external APIs."""
from hashlib import sha256
from math import sqrt
from typing import Sequence
from pdf_rag.domain.models import (
    ABSTENTION_TEXT, AnswerDraft, AnswerStatus, DocumentChunk, DocumentPage,
    GenerationResult, GroundingContext, GroundingEvidence, RawDocument,
    RetrievalHit, RetrieverResult, SourceCitation, SynthesizedAnswer,
    TokenUsage, VectorIndex,
)
from pdf_rag.application.config import IndexBuildConfig
from pdf_rag.application.ports import VectorMatch


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def document(name="research.pdf", texts=("Page one evidence.",)) -> RawDocument:
    raw_hash = digest(name + repr(texts))
    return RawDocument(
        document_id=digest(f"document:{name}:{raw_hash}"),
        file_name=name, content_sha256=raw_hash,
        file_size_bytes=1024, page_count=len(texts),
        pages=tuple(DocumentPage(page_number=i, text=t) for i, t in enumerate(texts, 1)),
    )


def chunk(name="research.pdf", text="Page one evidence.", *, page=1, index=0, doc_id=None) -> DocumentChunk:
    # Test-fixture repair: text changes must not create a new parent document ID.
    # Previously every chunk of the same file got a different document ID.
    identifier = doc_id or digest(f"fixture-document:{name}")
    return DocumentChunk(
        chunk_id=digest(f"chunk:{identifier}:{page}:{index}:{text}"),
        document_id=identifier, file_name=name, page_number=page,
        chunk_index=index, text=text,
    )


def chunks_for(doc: RawDocument) -> tuple[DocumentChunk, ...]:
    return tuple(
        chunk(doc.file_name, page.text, page=page.page_number, doc_id=doc.document_id)
        for page in doc.pages if page.text.strip()
    )


def config() -> IndexBuildConfig:
    return IndexBuildConfig(embedding_profile_id=digest("embed-profile-v1"))


def index(chunks: Sequence[DocumentChunk]) -> VectorIndex:
    return VectorIndex(
        index_id=digest("index" + "".join(c.chunk_id for c in chunks)),
        embedding_model="gemini-embedding-2", embedding_dimension=768,
        embedding_profile_id=digest("embed-profile-v1"),
        chunk_count=len(chunks), document_count=len({c.document_id for c in chunks}),
    )


def results(chunks: Sequence[DocumentChunk], *, question="What is the topic?", scores=None, threshold=0.70):
    actual_scores = scores or [0.99 - 0.01 * i for i in range(len(chunks))]
    hits = tuple(
        RetrievalHit(chunk=c, rank=i+1, similarity_score=s)
        for i, (c, s) in enumerate(zip(chunks, actual_scores))
    )
    return RetrieverResult(
        query_text=question, index_id=index(chunks).index_id,
        hits=hits, candidate_k=max(5, len(chunks)), similarity_threshold=threshold,
    )


def grounded(chunks: Sequence[DocumentChunk]) -> GroundingContext:
    entries = tuple(
        GroundingEvidence(
            citation_id=f"S{i}", chunk_id=c.chunk_id, document_id=c.document_id,
            file_name=c.file_name, page_number=c.page_number, text=c.text,
        ) for i, c in enumerate(chunks, start=1)
    )
    return GroundingContext(
        context_text="\n".join(f"[{e.citation_id}] {e.file_name} page {e.page_number}: {e.text}" for e in entries),
        evidence=entries,
        allowed_citation_ids=tuple(e.citation_id for e in entries),
    )


def valid_answer(c: DocumentChunk, answer="The result is 42. [S1]") -> SynthesizedAnswer:
    return SynthesizedAnswer(
        answer_text=answer, status=AnswerStatus.ANSWERED,
        citations=(SourceCitation(citation_id="S1", chunk_id=c.chunk_id, document_id=c.document_id,
                                file_name=c.file_name, page_number=c.page_number),),
        token_usage=TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30),
    )


def no_answer() -> SynthesizedAnswer:
    return SynthesizedAnswer(answer_text=ABSTENTION_TEXT, status=AnswerStatus.INSUFFICIENT_EVIDENCE)


class FakeEmbeddingProvider:
    """Configurable vectors; records exactly how many document and query API calls occur."""
    def __init__(self, vectors: dict[str, Sequence[float]], dimension: int = 768):
        self.vectors = vectors
        self.dimension = dimension
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []
        self.error = None

    def _vector(self, text):
        small = list(self.vectors[text])
        return small + [0.0] * (self.dimension - len(small)) if len(small) <= self.dimension else small

    def embed_documents(self, texts):
        self.document_calls.append(tuple(texts))
        if self.error:
            raise self.error
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        if self.error:
            raise self.error
        return self._vector(text)


class FakeVectorBackend:
    """Plain dot-product fake: manager is responsible for normalizing vectors."""
    def __init__(self, vectors):
        self.vectors = [list(v) for v in vectors]
        self.query_calls = []

    def search(self, query_vector, *, top_k):
        self.query_calls.append((list(query_vector), top_k))
        scores = [sum(a*b for a, b in zip(row, query_vector)) for row in self.vectors]
        ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [VectorMatch(position=i, similarity_score=scores[i]) for i in ordered[:top_k]]


class FakeVectorFactory:
    def __init__(self):
        self.built_vectors = []
        self.backends = []

    def build(self, vectors):
        self.built_vectors.append([list(v) for v in vectors])
        backend = FakeVectorBackend(vectors)
        self.backends.append(backend)
        return backend


class FakeGenerator:
    def __init__(self, result=None, error=None):
        self.result = result or GenerationResult(
            draft=AnswerDraft(answer_text="The result is 42. [S1]", status=AnswerStatus.ANSWERED,
                              citation_ids=("S1",)),
            token_usage=TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        self.error = error
        self.calls = []

    def generate(self, question, context, *, policy):
        self.calls.append((question, context, policy))
        if self.error:
            raise self.error
        return self.result
