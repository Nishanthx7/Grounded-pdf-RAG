import pytest
from pydantic import ValidationError
from tests._builders import chunk, document, digest, grounded, index, no_answer, valid_answer
from pdf_rag.domain.models import (
    ABSTENTION_TEXT, AnswerStatus, DocumentChunk, DocumentPage, GroundingContext,
    GroundingEvidence, IngestionReport, RawDocument, RetrievalHit, RetrieverResult,
    RejectedUpload, SourceCitation, SynthesizedAnswer, TokenUsage, VectorIndex,
)


def test_raw_document_accepts_original_numbered_pages():
    d = document(texts=("First", "", "Third"))
    assert d.page_count == 3
    assert [p.page_number for p in d.pages] == [1, 2, 3]

@pytest.mark.parametrize("pages", [
    (DocumentPage(page_number=2, text="text"),),
    (DocumentPage(page_number=1, text=" " ),),
    (DocumentPage(page_number=1, text="a"), DocumentPage(page_number=1, text="b")),
])
def test_raw_document_rejects_invalid_page_sequence_or_empty_content(pages):
    with pytest.raises(ValidationError):
        RawDocument(document_id=digest("doc"), file_name="a.pdf", content_sha256=digest("pdf"),
                    file_size_bytes=25, page_count=len(pages), pages=pages)

@pytest.mark.parametrize("file_name", ["../private.pdf", "nested\\test.pdf"])
def test_raw_document_disallows_directory_components(file_name):
    with pytest.raises(ValidationError):
        RawDocument(**(document().model_dump() | {"file_name": file_name}))


def test_models_are_immutable_and_forbid_unknown_fields():
    d = document()
    with pytest.raises(ValidationError):
        d.file_name = "change.pdf"
    with pytest.raises(ValidationError):
        DocumentPage(page_number=1, text="a", unexpected="bad")

@pytest.mark.parametrize("value", ["", "\t", "\n  "])
def test_chunk_rejects_whitespace_only_text(value):
    with pytest.raises(ValidationError):
        DocumentChunk(**(chunk().model_dump() | {"text": value}))


def test_vector_index_rejects_more_documents_than_chunks():
    v = index([chunk()])
    with pytest.raises(ValidationError):
        VectorIndex(**(v.model_dump() | {"document_count": 2}))


def test_vector_index_requires_cosine_and_normalization():
    v = index([chunk()])
    for invalid in ({"similarity_metric": "l2"}, {"normalization_enabled": False}):
        with pytest.raises(ValidationError):
            VectorIndex(**(v.model_dump() | invalid))


def test_retriever_result_accepts_no_hits():
    r = RetrieverResult(query_text="Question", index_id=digest("idx"), candidate_k=5,
                        similarity_threshold=0.7)
    assert r.hits == ()

@pytest.mark.parametrize("modification", [
    "below_threshold", "wrong_rank", "duplicate", "wrong_order", "too_many",
])
def test_retrieval_invariants(modification):
    c1, c2 = chunk(text="first"), chunk(text="second")
    hits = (RetrievalHit(chunk=c1, rank=1, similarity_score=.95),
            RetrievalHit(chunk=c2, rank=2, similarity_score=.80))
    if modification == "below_threshold":
        hits = (RetrievalHit(chunk=c1, rank=1, similarity_score=.4),)
    elif modification == "wrong_rank":
        hits = (RetrievalHit(chunk=c1, rank=2, similarity_score=.9),)
    elif modification == "duplicate":
        hits = (hits[0], RetrievalHit(chunk=c1, rank=2, similarity_score=.8))
    elif modification == "wrong_order":
        hits = (RetrievalHit(chunk=c1, rank=1, similarity_score=.8),
                RetrievalHit(chunk=c2, rank=2, similarity_score=.95))
    with pytest.raises(ValidationError):
        RetrieverResult(query_text="Q", index_id=digest("index"), hits=hits,
                        candidate_k=1 if modification == "too_many" else 5,
                        similarity_threshold=.7)


def test_grounding_requires_matching_unique_citation_ids():
    good = grounded((chunk(),))
    with pytest.raises(ValidationError):
        GroundingContext(**(good.model_dump() | {"allowed_citation_ids": ("S2",)}))
    with pytest.raises(ValidationError):
        GroundingContext(**(good.model_dump() | {"evidence": good.evidence + good.evidence}))


def test_answered_requires_citation():
    with pytest.raises(ValidationError):
        SynthesizedAnswer(answer_text="Answer", status=AnswerStatus.ANSWERED)


def test_abstention_requires_canonical_message_and_zero_citations():
    assert no_answer().answer_text == ABSTENTION_TEXT
    with pytest.raises(ValidationError):
        SynthesizedAnswer(answer_text="maybe", status=AnswerStatus.INSUFFICIENT_EVIDENCE)
    with pytest.raises(ValidationError):
        SynthesizedAnswer(answer_text=ABSTENTION_TEXT, status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                          citations=valid_answer(chunk()).citations)


def test_answer_rejects_duplicate_citations():
    answer = valid_answer(chunk())
    with pytest.raises(ValidationError):
        SynthesizedAnswer(**(answer.model_dump() | {"citations": answer.citations * 2}))


def test_unknown_token_usage_is_not_zero():
    usage = TokenUsage()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=-1)


def test_ingestion_report_rejects_orphan_or_mismatched_chunk():
    d = document()
    with pytest.raises(ValidationError):
        IngestionReport(raw_documents=(d,), chunks=(chunk(doc_id=digest("orphan")),))
    c = chunk(doc_id=d.document_id, name=d.file_name)
    with pytest.raises(ValidationError):
        IngestionReport(raw_documents=(d,), chunks=(DocumentChunk(**(c.model_dump() | {"page_number": 9})),))


def test_ingestion_report_rejects_duplicate_chunk_identity():
    d = document()
    c = chunk(doc_id=d.document_id)
    with pytest.raises(ValidationError):
        IngestionReport(raw_documents=(d,), chunks=(c, c))
