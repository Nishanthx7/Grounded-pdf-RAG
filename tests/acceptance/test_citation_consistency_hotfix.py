"""Targeted regressions for redundant citation representations in Gemini drafts."""
from unittest.mock import Mock
import pytest
from pdf_rag.application.config import GenerationPolicy
from pdf_rag.domain.models import AnswerDraft, AnswerStatus, GenerationResult
from pdf_rag.domain.errors import AnswerValidationError
from pdf_rag.infrastructure.gemini.adapters import GeminiStructuredAnswerGenerator
from tests._builders import chunk, grounded

CONTEXT = grounded((chunk(text="LLMs focus on language", page=4), chunk(text="GenAI creates multiple media", page=5)))

def invoke(draft: AnswerDraft):
    chain = Mock()
    chain.invoke.return_value = GenerationResult(draft=draft)
    adapter = GeminiStructuredAnswerGenerator(chain)
    return adapter.generate("Compare LLM and GenAI", CONTEXT, policy=GenerationPolicy()).draft


def test_missing_inline_markers_are_added_from_allowed_structured_ids():
    draft = invoke(AnswerDraft(answer_text="An LLM works with language; GenAI spans other modalities.", status=AnswerStatus.ANSWERED, citation_ids=("S1", "S2")))
    assert draft.answer_text.endswith("[S1] [S2]")
    assert draft.citation_ids == ("S1", "S2")


def test_missing_structured_ids_are_derived_from_valid_inline_ids():
    draft = invoke(AnswerDraft(answer_text="An LLM works with language [S1]. GenAI creates content [S2] and pictures [S2].", status=AnswerStatus.ANSWERED, citation_ids=()))
    assert draft.citation_ids == ("S1", "S2")
    assert draft.answer_text == "An LLM works with language [S1]. GenAI creates content [S2] and pictures [S2]."


def test_explicit_disagreement_remains_unmodified():
    original = AnswerDraft(answer_text="Text says this [S2].", status=AnswerStatus.ANSWERED, citation_ids=("S1",))
    assert invoke(original) == original


def test_unprovided_structured_ids_cannot_be_repaired():
    original = AnswerDraft(answer_text="Claim", status=AnswerStatus.ANSWERED, citation_ids=("S99",))
    assert invoke(original) == original


def test_unprovided_inline_ids_cannot_be_repaired():
    original = AnswerDraft(answer_text="Claim [S99]", status=AnswerStatus.ANSWERED, citation_ids=())
    assert invoke(original) == original


def test_empty_both_still_invalid():
    original = AnswerDraft(answer_text="Claim", status=AnswerStatus.ANSWERED, citation_ids=())
    assert invoke(original) == original


def test_abstention_unchanged():
    original = AnswerDraft(answer_text="Insufficient evidence", status=AnswerStatus.INSUFFICIENT_EVIDENCE, citation_ids=())
    assert invoke(original) == original


def test_duplicate_structured_ids_not_repaired():
    original = AnswerDraft(answer_text="Claim", status=AnswerStatus.ANSWERED, citation_ids=("S1", "S1"))
    assert invoke(original) == original


def test_malformed_inline_marker_is_not_automatically_blindly_certified():
    original = AnswerDraft(answer_text="Claim [S01]", status=AnswerStatus.ANSWERED, citation_ids=("S1",))
    assert invoke(original) == original


def test_pipeline_accepts_generator_repaired_structured_omission():
    from pdf_rag.application.services import DefaultRAGPipeline
    from pdf_rag.application.config import RetrievalPolicy
    from tests._builders import index, results
    chunks = (chunk(text="Language models handle language", page=4),)
    idx = index(chunks)
    store = Mock()
    store.retrieve.return_value = results(chunks).model_copy(update={"index_id": idx.index_id})
    chain = Mock()
    chain.invoke.return_value = GenerationResult(draft=AnswerDraft(
        answer_text="LLMs handle language [S1].", status=AnswerStatus.ANSWERED,
        citation_ids=(),
    ))
    generator = GeminiStructuredAnswerGenerator(chain)
    answer = DefaultRAGPipeline(store, generator).answer(
        idx, "What are LLMs?", retrieval=RetrievalPolicy(),
        generation=GenerationPolicy(),
    )
    assert answer.status == AnswerStatus.ANSWERED
    assert answer.citations[0].page_number == 4
    assert answer.answer_text == "LLMs handle language [S1]."


def test_pipeline_still_rejects_conflicting_explicit_citations(caplog):
    from pdf_rag.application.services import DefaultRAGPipeline
    from pdf_rag.application.config import RetrievalPolicy
    from tests._builders import index, results
    cs = (chunk(text="The first source", page=4), chunk(text="The second source", page=5))
    idx = index(cs)
    store = Mock()
    store.retrieve.return_value = results(cs).model_copy(update={"index_id": idx.index_id})
    chain = Mock()
    chain.invoke.return_value = GenerationResult(draft=AnswerDraft(
        answer_text="Evidence [S1].", status=AnswerStatus.ANSWERED,
        citation_ids=("S2",),
    ))
    pipeline = DefaultRAGPipeline(store, GeminiStructuredAnswerGenerator(chain))
    with pytest.raises(AnswerValidationError, match="Text citations and structured citations disagree"):
        pipeline.answer(idx, "Question", retrieval=RetrievalPolicy(), generation=GenerationPolicy())
    assert "Citation mismatch: inline_ids=" in caplog.text
    assert "The first source" not in caplog.text
    assert "The second source" not in caplog.text
