"""Thin Streamlit UI for session-scoped, strictly PDF-grounded question answering.

No PDF parser, vector backend or Gemini client is imported here. The caller
injects application services, allowing deterministic tests without Streamlit.
"""
from __future__ import annotations

from hashlib import sha256
import json
import logging
from typing import Any, Sequence

from pydantic import ValidationError

from pdf_rag.application.config import (
    ChunkingPolicy, GenerationPolicy, IndexBuildConfig, IngestionPolicy, RetrievalPolicy,
)
from pdf_rag.application.contracts import DocumentProcessor, VectorStoreManager, RAGPipeline
from pdf_rag.domain.errors import (
    RAGError, ConfigurationError, DocumentValidationError, DocumentExtractionError,
    ChunkingError, EmbeddingError, IndexBuildError, IndexNotFoundError,
    RetrievalError, ContextAssemblyError, AnswerGenerationError,
    AnswerValidationError, RateLimitError, QuotaExhaustedError,
)
from pdf_rag.domain.models import AnswerStatus, IngestionReport, PdfUpload, SourceCitation

LOG = logging.getLogger(__name__)

# Only enumerate controlled validation messages produced by our application.
# Never log the raw LLM response, PDF text, provider exception or API key.
_SAFE_VALIDATION_REASONS = frozenset({
    "Gemini response failed structured output parsing",
    "Gemini returned an invalid answer schema",
    "Question must not be blank",
    "An abstention cannot cite sources",
    "Unknown generated answer status",
    "Answer requires unique citation IDs",
    "Answer references unprovided evidence",
    "Text citations and structured citations disagree",
})


def _safe_validation_reason(exc: Exception) -> str:
    if not isinstance(exc, AnswerValidationError):
        return "not_applicable"
    description = str(exc)
    return description if description in _SAFE_VALIDATION_REASONS else "other_validation_error"


ERROR_MESSAGES: tuple[tuple[type[Exception], str], ...] = (
    (QuotaExhaustedError, "Gemini's API quota has been exhausted. Check your quota and retry later."),
    (RateLimitError, "Gemini is rate-limiting requests. Please retry shortly."),
    (ConfigurationError, "Gemini configuration or API access is invalid. Check your API key and model settings."),
    (DocumentValidationError, "The PDF upload violates the configured file or page limits."),
    (DocumentExtractionError, "Unable to extract usable text from the selected PDF."),
    (ChunkingError, "PDF text splitting failed. Check the input PDF and configuration."),
    (EmbeddingError, "Could not generate Gemini embeddings. Check API access and quota."),
    (IndexBuildError, "The vector index could not be constructed."),
    (IndexNotFoundError, "The PDF index is unavailable. Please reindex the PDFs."),
    (RetrievalError, "Vector retrieval failed. Please retry the question."),
    (ContextAssemblyError, "The selected PDF evidence exceeds the safe context budget."),
    (AnswerValidationError, "The model returned an invalid or unverified answer. No answer was saved."),
    (AnswerGenerationError, "Gemini could not generate an answer. Please retry."),
    (RAGError, "The document question-answering operation failed."),
    (ValidationError, "The input or returned data failed validation."),
)


def _safe_error(exc: Exception) -> str:
    """Never display provider/raw exception bodies, which may contain secrets."""
    for kind, message in ERROR_MESSAGES:
        if isinstance(exc, kind):
            return message
    return "An unexpected application error occurred. See safe diagnostic logs."


def _selection_fingerprint(uploads: Sequence[PdfUpload], *, ingestion: IngestionPolicy,
                           chunking: ChunkingPolicy, index: IndexBuildConfig) -> str:
    """Stable across uploader reordering; sensitive to selected bytes and index config."""
    files = sorted((u.file_name, sha256(u.content).hexdigest()) for u in uploads)
    payload = {
        "version": 1,
        "files": files,
        "ingestion": ingestion.model_dump(),
        "chunking": chunking.model_dump(),
        "index": index.model_dump(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


class DefaultStreamlitApp:
    """Presentation adapter; all working index resources stay inside session.

    The same VectorStoreManager object must be injected across Streamlit reruns.
    `app.py` enforces this through a session-scoped composition root.
    """

    def __init__(
        self,
        processor: DocumentProcessor,
        vector_manager: VectorStoreManager,
        pipeline: RAGPipeline,
        *,
        ui: Any,
        ingestion: IngestionPolicy | None = None,
        chunking: ChunkingPolicy | None = None,
        index_config: IndexBuildConfig | None = None,
        retrieval: RetrievalPolicy | None = None,
        generation: GenerationPolicy | None = None,
    ) -> None:
        self.processor = processor
        self.vector_manager = vector_manager
        self.pipeline = pipeline
        self.ui = ui
        self.ingestion = ingestion or IngestionPolicy()
        self.chunking = chunking or ChunkingPolicy()
        if index_config is None:
            # For separately injected fake managers in acceptance tests only.
            # Production bootstrap always injects the provider-specific profile.
            from pdf_rag.infrastructure.gemini.adapters import embedding_profile_id
            index_config = IndexBuildConfig(embedding_profile_id=embedding_profile_id())
        self.index_config = index_config
        self.retrieval = retrieval or RetrievalPolicy()
        self.generation = generation or GenerationPolicy()

    def _initial_state(self) -> None:
        state = self.ui.session_state
        defaults: dict[str, Any] = {
            "vector_index": None,
            "chat_history": [],
            "upload_fingerprint": None,
            "index_status": "empty",
            "ingestion_report": None,
            "source_excerpts": {},
            "index_error": None,
            "upload_widget_version": 0,
        }
        for key, value in defaults.items():
            if key not in state:
                state[key] = value

    def _clear_index(self, *, clear_history: bool = True) -> None:
        state = self.ui.session_state
        self.vector_manager.clear()
        state["vector_index"] = None
        state["ingestion_report"] = None
        state["source_excerpts"] = {}
        state["index_error"] = None
        state["index_status"] = "empty"
        if clear_history:
            state["chat_history"] = []

    def _reset(self) -> None:
        state = self.ui.session_state
        self._clear_index()
        state["upload_fingerprint"] = None
        # Streamlit prohibits programmatic mutation of an instantiated file uploader;
        # changing its key on this rerun is the supported reset mechanism.
        state["upload_widget_version"] += 1

    def _process_selection(self, uploads: Sequence[PdfUpload], fingerprint: str) -> None:
        state = self.ui.session_state
        # Invalidate stale evidence *before* PDF processing or API requests.
        self._clear_index()
        state["upload_fingerprint"] = fingerprint
        state["index_status"] = "validating"
        progress = self.ui.progress(0, text="Validating PDFs and extracting text...")
        try:
            report = self.processor.process_uploads(
                uploads, ingestion=self.ingestion, chunking=self.chunking,
            )
            state["ingestion_report"] = report
            state["source_excerpts"] = {c.chunk_id: c.text for c in report.chunks}
            progress.progress(45, text="PDF extraction and chunking complete")
            if not report.chunks:
                state["index_status"] = "failed"
                state["index_error"] = "No searchable text was found in the selected PDFs."
                return
            state["index_status"] = "indexing"
            progress.progress(60, text="Embedding chunks and building the FAISS index...")
            new_index = self.vector_manager.build_index(report.chunks, config=self.index_config)
            # Commit only after the manager has published a complete ready index.
            state["vector_store"] = self.vector_manager
            state["vector_index"] = new_index
            state["index_status"] = "ready"
            state["index_error"] = None
            progress.progress(100, text="Index ready")
        except Exception as exc:
            LOG.warning("Indexing failed (%s)", type(exc).__name__)
            # Preserve the rejected file report if processing got that far.
            self.vector_manager.clear()
            state["vector_index"] = None
            state["index_status"] = "failed"
            state["index_error"] = _safe_error(exc)
            self.ui.error(state["index_error"])
        finally:
            progress.empty()

    def _display_report(self) -> None:
        state = self.ui.session_state
        report: IngestionReport | None = state.get("ingestion_report")
        if report is None:
            return
        if report.raw_documents:
            self.ui.caption(
                f"Indexed selection: {len(report.raw_documents)} PDF(s), "
                f"{len(report.chunks)} text chunks"
            )
        for rejection in report.rejected_files:
            # Rejection messages originate from the PDF processor, not user-supplied text.
            self.ui.warning(f"File rejected: {rejection.file_name} — {rejection.message}")
        for document in report.raw_documents:
            for issue in document.extraction_issues:
                self.ui.warning(f"Partial extraction: {document.file_name}, page {issue.page_number}")

    def _display_chat(self) -> None:
        for message in self.ui.session_state["chat_history"]:
            # The history is an ordered transcript for display only, never prompt memory.
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                continue
            with self.ui.chat_message(message["role"]):
                self.ui.markdown(message.get("content", ""))
                if message["role"] == "assistant":
                    self._display_sources(message.get("sources", []))

    def _display_sources(self, sources: Sequence[dict[str, Any]]) -> None:
        for source in sources:
            # The source label is application-authoritative, never an LLM citation string.
            label = f"Source [{source['citation_id']}]: {source['file_name']} · page {source['page_number']}"
            with self.ui.expander(label):
                self.ui.write(f"PDF: {source['file_name']}")
                self.ui.write(f"Page: {source['page_number']}")
                self.ui.caption(f"Chunk ID: {source['chunk_id']}")
                excerpt = self.ui.session_state.get("source_excerpts", {}).get(source["chunk_id"])
                if excerpt:
                    self.ui.text(excerpt)

    def _answer_question(self, question: str) -> None:
        state = self.ui.session_state
        if not question.strip() or state.get("index_status") != "ready":
            return
        # Independent-question RAG: previous history is NEVER included in the call.
        state["chat_history"].append({"role": "user", "content": question})
        with self.ui.chat_message("user"):
            self.ui.markdown(question)
        with self.ui.chat_message("assistant"):
            try:
                with self.ui.spinner("Finding evidence and generating an answer..."):
                    result = self.pipeline.answer(
                        state["vector_index"], question,
                        retrieval=self.retrieval, generation=self.generation,
                    )
                self.ui.markdown(result.answer_text)
                sources = [c.model_dump() for c in result.citations]
                self._display_sources(sources)
                state["chat_history"].append({
                    "role": "assistant",
                    "content": result.answer_text,
                    "status": result.status.value,
                    "sources": sources,
                })
            except Exception as exc:
                LOG.warning(
                    "Question answering failed (%s), validation_reason=%s",
                    type(exc).__name__, _safe_validation_reason(exc),
                )
                self.ui.error(_safe_error(exc))
                # No fabricated assistant answer is added to the session transcript.

    def run(self) -> None:
        ui = self.ui
        self._initial_state()
        state = ui.session_state
        ui.title("PDF Question Answering")
        ui.caption("Answers are grounded in uploaded text PDFs. Gemini processes uploaded text in the cloud.")

        # Sidebar reset happens BEFORE rendering the uploader so its widget key
        # can be rotated without mutating Streamlit's protected widget state.
        sidebar = ui.sidebar
        sidebar.markdown("### Document workspace")
        if sidebar.button("Reset documents and chat", key="reset_corpus") is True:
            self._reset()
        retry = sidebar.button("Retry indexing", key="retry_index", disabled=state["index_status"] != "failed") is True
        sidebar.caption("Up to 10 text PDFs · 20 MB/file · 100 MB total · 300 pages")
        sidebar.caption("Session-only index; refreshed pages or server restarts lose the index.")
        sidebar.caption("Avoid confidential PDFs when using a free-tier Gemini API account.")

        chosen = ui.file_uploader(
            "Upload your PDF documents", type=["pdf"], accept_multiple_files=True,
            key=f"pdf_upload_{state['upload_widget_version']}",
        ) or []
        # getvalue() is non-destructive; no UploadedFile objects are stored in state.
        # Invalid uploader metadata must never leave a previously ready corpus active.
        try:
            uploads = tuple(PdfUpload(file_name=f.name, content=f.getvalue()) for f in chosen)
        except Exception as exc:
            LOG.warning("Upload conversion failed (%s)", type(exc).__name__)
            self._clear_index()
            state["upload_fingerprint"] = None
            state["index_status"] = "failed"
            state["index_error"] = _safe_error(exc)
            ui.error(state["index_error"])
            return
        if not uploads:
            if state.get("upload_fingerprint") is not None:
                self._clear_index()
                state["upload_fingerprint"] = None
        else:
            fingerprint = _selection_fingerprint(
                uploads, ingestion=self.ingestion, chunking=self.chunking,
                index=self.index_config,
            )
            if fingerprint != state["upload_fingerprint"] or retry:
                self._process_selection(uploads, fingerprint)

        self._display_report()
        if state["index_status"] == "failed" and state.get("index_error"):
            ui.info(state["index_error"] + " Select Retry indexing to try again.")
        if state["index_status"] == "ready":
            ui.success("Documents indexed. Ask a question about their contents.")
        elif not uploads:
            ui.info("Upload one or more text PDFs to begin.")

        self._display_chat()
        question = ui.chat_input(
            "Ask a question about the PDFs",
            disabled=state["index_status"] != "ready",
        )
        if isinstance(question, str) and question.strip():
            self._answer_question(question)
