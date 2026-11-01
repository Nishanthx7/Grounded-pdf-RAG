"""Step 3 immutable domain contracts, unchanged for Step 4."""
from enum import Enum
from typing import Annotated, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CitationId = Annotated[str, Field(pattern=r"^S[1-9][0-9]*$")]

class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class DocumentPage(DomainModel):
    page_number: int = Field(ge=1)
    text: str

class ExtractionIssue(DomainModel):
    page_number: int = Field(ge=1)
    message: str = Field(min_length=1)

class RawDocument(DomainModel):
    document_id: Sha256
    file_name: str = Field(min_length=1)
    content_sha256: Sha256
    file_size_bytes: int = Field(gt=0)
    page_count: int = Field(gt=0)
    pages: tuple[DocumentPage, ...] = Field(min_length=1)
    extraction_issues: tuple[ExtractionIssue, ...] = ()

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        page_numbers = [p.page_number for p in self.pages]
        if page_numbers != list(range(1, self.page_count + 1)):
            raise ValueError("Pages must preserve original PDF order")
        if not any(page.text.strip() for page in self.pages):
            raise ValueError("Document has no extractable text")
        if "/" in self.file_name or "\\" in self.file_name:
            raise ValueError("file_name must be sanitized")
        if any(issue.page_number > self.page_count for issue in self.extraction_issues):
            raise ValueError("Extraction issue references invalid page")
        return self

class DocumentChunk(DomainModel):
    chunk_id: Sha256
    document_id: Sha256
    file_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chunk(self) -> Self:
        if not self.text.strip():
            raise ValueError("Chunk cannot be whitespace-only")
        return self

class VectorIndex(DomainModel):
    index_id: Sha256
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    embedding_profile_id: Sha256
    similarity_metric: Literal["cosine"] = "cosine"
    normalization_enabled: Literal[True] = True
    chunk_count: int = Field(gt=0)
    document_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.document_count > self.chunk_count:
            raise ValueError("Invalid index document count")
        return self

class RetrievalHit(DomainModel):
    chunk: DocumentChunk
    rank: int = Field(ge=1)
    similarity_score: float = Field(ge=-1.0, le=1.0)

class RetrieverResult(DomainModel):
    query_text: str = Field(min_length=1)
    index_id: Sha256
    hits: tuple[RetrievalHit, ...] = ()
    candidate_k: int = Field(gt=0)
    similarity_threshold: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_results(self) -> Self:
        if not self.query_text.strip():
            raise ValueError("Question cannot be blank")
        if len(self.hits) > self.candidate_k:
            raise ValueError("Too many retrieval hits")
        seen: set[str] = set()
        for position, hit in enumerate(self.hits, start=1):
            if hit.rank != position:
                raise ValueError("Retrieval ranks must be consecutive")
            if hit.similarity_score < self.similarity_threshold:
                raise ValueError("Hit is below similarity threshold")
            if hit.chunk.chunk_id in seen:
                raise ValueError("Duplicate retrieval chunk")
            seen.add(hit.chunk.chunk_id)
        ordered = sorted(self.hits, key=lambda hit: (-hit.similarity_score, hit.chunk.chunk_id))
        if list(self.hits) != ordered:
            raise ValueError("Hits must be deterministically ranked")
        return self

class GroundingEvidence(DomainModel):
    citation_id: CitationId
    chunk_id: Sha256
    document_id: Sha256
    file_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)

class GroundingContext(DomainModel):
    context_text: str = Field(min_length=1)
    evidence: tuple[GroundingEvidence, ...] = Field(min_length=1)
    allowed_citation_ids: tuple[CitationId, ...] = Field(min_length=1)
    context_token_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        expected = tuple(item.citation_id for item in self.evidence)
        if len(set(expected)) != len(expected):
            raise ValueError("Duplicate evidence citation IDs")
        if expected != self.allowed_citation_ids:
            raise ValueError("Citation IDs do not match evidence")
        if not self.context_text.strip():
            raise ValueError("Grounding context cannot be blank")
        return self

class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

class SourceCitation(DomainModel):
    citation_id: CitationId
    chunk_id: Sha256
    document_id: Sha256
    file_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)

class TokenUsage(DomainModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

ABSTENTION_TEXT = (
    "The uploaded PDFs do not contain sufficient "
    "evidence to answer this question."
)

class SynthesizedAnswer(DomainModel):
    answer_text: str = Field(min_length=1)
    status: AnswerStatus
    citations: tuple[SourceCitation, ...] = ()
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def validate_answer(self) -> Self:
        if not self.answer_text.strip():
            raise ValueError("Answer cannot be blank")
        if self.status == AnswerStatus.ANSWERED:
            if not self.citations:
                raise ValueError("Answered result requires citations")
        if self.status == AnswerStatus.INSUFFICIENT_EVIDENCE:
            if self.citations:
                raise ValueError("Abstention cannot contain citations")
            if self.answer_text != ABSTENTION_TEXT:
                raise ValueError("Invalid abstention message")
        ids = [citation.citation_id for citation in self.citations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate citations")
        return self

class PdfUpload(DomainModel):
    file_name: str = Field(min_length=1)
    content: bytes = Field(repr=False)

class RejectedUpload(DomainModel):
    file_name: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)

class IngestionReport(DomainModel):
    raw_documents: tuple[RawDocument, ...] = ()
    chunks: tuple[DocumentChunk, ...] = ()
    rejected_files: tuple[RejectedUpload, ...] = ()

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        documents = {doc.document_id: doc for doc in self.raw_documents}
        if len(documents) != len(self.raw_documents):
            raise ValueError("Duplicate document identities")
        seen_chunks: set[str] = set()
        for chunk in self.chunks:
            parent = documents.get(chunk.document_id)
            if parent is None:
                raise ValueError("Chunk has no parent document")
            if chunk.file_name != parent.file_name:
                raise ValueError("Chunk filename does not match parent")
            if chunk.page_number > parent.page_count:
                raise ValueError("Chunk references invalid page")
            if not parent.pages[chunk.page_number - 1].text.strip():
                raise ValueError("Chunk references empty source page")
            if chunk.chunk_id in seen_chunks:
                raise ValueError("Duplicate chunk identity")
            seen_chunks.add(chunk.chunk_id)
        return self

class AnswerDraft(DomainModel):
    answer_text: str = Field(min_length=1)
    status: AnswerStatus
    citation_ids: tuple[CitationId, ...] = ()

class GenerationResult(DomainModel):
    draft: AnswerDraft
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
