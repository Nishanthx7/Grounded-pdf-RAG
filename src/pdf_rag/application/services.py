"""Application orchestration. No direct Google, FAISS, or Streamlit imports."""
from __future__ import annotations

import re
import logging
from pdf_rag.application.contracts import VectorStoreManager
from pdf_rag.application.ports import StructuredAnswerGenerator
from pdf_rag.application.config import GenerationPolicy, RetrievalPolicy
from pdf_rag.application.grounding import GroundingContextAssembler
from pdf_rag.domain.errors import (
    AnswerGenerationError, AnswerValidationError, ContextAssemblyError,
    RAGError, RetrievalError,
)
from pdf_rag.domain.models import (
    ABSTENTION_TEXT, AnswerStatus, SourceCitation, SynthesizedAnswer,
    TokenUsage, VectorIndex,
)

# Preserve Step 4 compatibility import paths.
from pdf_rag.infrastructure.pdf.processor import DefaultDocumentProcessor
from pdf_rag.infrastructure.vector.manager import SessionVectorStoreManager

LOG = logging.getLogger(__name__)
_INLINE_REFERENCE = re.compile(r"\[(S[0-9]+)\]")


def _abstention(usage: TokenUsage | None = None) -> SynthesizedAnswer:
    return SynthesizedAnswer(
        answer_text=ABSTENTION_TEXT,
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        token_usage=usage if usage is not None else TokenUsage(),
    )


class DefaultRAGPipeline:
    """Retrieval -> bounded evidence -> untrusted draft -> validated answer."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        generator: StructuredAnswerGenerator,
        *,
        assembler: GroundingContextAssembler | None = None,
    ) -> None:
        self._vector_store_manager = vector_store_manager
        self._generator = generator
        self._assembler = assembler or GroundingContextAssembler()

    def answer(
        self, index: VectorIndex, question: str, *,
        retrieval: RetrievalPolicy, generation: GenerationPolicy,
    ) -> SynthesizedAnswer:
        if not isinstance(question, str) or not question.strip():
            raise RetrievalError("Question cannot be blank")

        # Do not swallow typed infrastructure errors or confuse them with
        # insufficient evidence.
        result = self._vector_store_manager.retrieve(index, question, policy=retrieval)
        if result.index_id != index.index_id:
            raise RetrievalError("Retrieval returned an inconsistent index")
        if not result.hits:
            return _abstention()   # No LLM call.

        context = self._assembler.assemble(
            result, context_token_budget=generation.context_token_budget,
        )
        try:
            generation_result = self._generator.generate(question, context, policy=generation)
        except RAGError:
            raise
        except Exception as exc:
            raise AnswerGenerationError("Structured answer generation failed") from None

        draft = generation_result.draft
        if draft.status == AnswerStatus.INSUFFICIENT_EVIDENCE:
            if draft.citation_ids or _INLINE_REFERENCE.search(draft.answer_text):
                raise AnswerValidationError("An abstention cannot cite sources")
            return _abstention(generation_result.token_usage)

        if draft.status != AnswerStatus.ANSWERED:
            raise AnswerValidationError("Unknown generated answer status")
        ids = tuple(draft.citation_ids)
        if not ids or len(set(ids)) != len(ids):
            raise AnswerValidationError("Answer requires unique citation IDs")
        if any(citation_id not in context.allowed_citation_ids for citation_id in ids):
            raise AnswerValidationError("Answer references unprovided evidence")
        in_text = tuple(_INLINE_REFERENCE.findall(draft.answer_text))
        if not in_text or set(in_text) != set(ids):
            LOG.warning("Citation mismatch: inline_ids=%s structured_ids=%s allowed_ids=%s",
                        tuple(sorted(set(in_text))), tuple(sorted(set(ids))),
                        context.allowed_citation_ids)
            raise AnswerValidationError("Text citations and structured citations disagree")
        # A repeated inline citation is legal (e.g. two sentences supported
        # by one source), but the structured list must be unique.
        refs = {entry.citation_id: entry for entry in context.evidence}
        citations = tuple(
            SourceCitation(
                citation_id=cid, chunk_id=refs[cid].chunk_id,
                document_id=refs[cid].document_id, file_name=refs[cid].file_name,
                page_number=refs[cid].page_number,
            ) for cid in ids
        )
        return SynthesizedAnswer(
            answer_text=draft.answer_text,
            status=AnswerStatus.ANSWERED,
            citations=citations,
            token_usage=generation_result.token_usage,
        )
