"""Page-preserving, bounded PDF ingestion using pypdf and LangChain splitters.

Only original uploaded bytes are read; nothing is written to disk. This is a
single-user local ingestion adapter, not an untrusted-PDF sandbox.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import logging
from pathlib import PurePosixPath
from typing import Sequence

from pypdf import PdfReader

from pdf_rag.application.config import ChunkingPolicy, IngestionPolicy
from pdf_rag.domain.errors import ChunkingError, DocumentValidationError
from pdf_rag.domain.models import (
    DocumentChunk,
    DocumentPage,
    ExtractionIssue,
    IngestionReport,
    PdfUpload,
    RawDocument,
    RejectedUpload,
)

_LOG = logging.getLogger(__name__)
_CHUNK_ID_VERSION = "pdf-rag-chunk-v1"
_DOCUMENT_ID_VERSION = b"pdf-rag-document-v1\x00"


def _safe_filename(name: str) -> str:
    """Produce a display-safe basename; reject names with control characters."""
    name = PurePosixPath(name.replace("\\", "/")).name.strip()
    if not name or name in {".", ".."} or any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        return ""
    return name


def _reject(name: str, code: str, message: str) -> RejectedUpload:
    return RejectedUpload(file_name=name or "unnamed", error_code=code, message=message)


def _document_id(name: str, content_hash: str) -> str:
    # Field boundaries and a schema version make identities unambiguous.
    digest = sha256()
    digest.update(_DOCUMENT_ID_VERSION)
    encoded_name = name.encode("utf-8")
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(bytes.fromhex(content_hash))
    return digest.hexdigest()


def _chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
    text: str,
    policy: ChunkingPolicy,
) -> str:
    payload = {
        "schema": _CHUNK_ID_VERSION,
        "document_id": document_id,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "text": text,
        "chunk_size": policy.chunk_size,
        "chunk_overlap": policy.chunk_overlap,
        "separators": list(policy.separators),
    }
    canonical_bytes = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(canonical_bytes).hexdigest()


def _new_splitter(policy: ChunkingPolicy):
    """Keep the optional library import at the infrastructure boundary."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ChunkingError(
            "langchain-text-splitters is required for PDF chunking; "
            "install the declared project dependencies."
        ) from exc

    return RecursiveCharacterTextSplitter(
        chunk_size=policy.chunk_size,
        chunk_overlap=policy.chunk_overlap,
        separators=list(policy.separators),
        length_function=len,
        keep_separator=True,
        is_separator_regex=False,
        strip_whitespace=True,
    )


class DefaultDocumentProcessor:
    """Convert accepted text PDFs into validated, page-scoped retrieval chunks.

    Batch limit failures abort processing. Individual invalid PDFs are reported
    in IngestionReport.rejected_files; no partial document is indexed.
    """

    def process_uploads(
        self,
        uploads: Sequence[PdfUpload],
        *,
        ingestion: IngestionPolicy,
        chunking: ChunkingPolicy,
    ) -> IngestionReport:
        # All batch-level limits are checked before doing any PDF parsing.
        if not uploads:
            raise DocumentValidationError("At least one PDF is required")
        if len(uploads) > ingestion.max_files:
            raise DocumentValidationError("Upload exceeds maximum file count")
        if sum(len(upload.content) for upload in uploads) > ingestion.max_total_size_bytes:
            raise DocumentValidationError("Upload exceeds aggregate byte limit")

        accepted: list[RawDocument] = []
        chunks: list[DocumentChunk] = []
        rejected: list[RejectedUpload] = []
        seen_ids: set[str] = set()
        total_pages = 0
        # Lazy: an entirely invalid batch never needs the text-splitter package.
        splitter = None

        for upload in uploads:
            name = _safe_filename(upload.file_name)
            if not name:
                rejected.append(_reject("unnamed", "invalid_filename", "Invalid PDF filename"))
                continue
            if not name.lower().endswith(".pdf") or not name[:-4].strip(" ."):
                rejected.append(_reject(name, "unsupported_file_type", "Only .pdf files are supported"))
                continue

            contents = upload.content
            size = len(contents)
            if size == 0:
                rejected.append(_reject(name, "empty_file", "The uploaded PDF is empty"))
                continue
            if size > ingestion.max_file_size_bytes:
                rejected.append(_reject(name, "file_too_large", "PDF exceeds the per-file byte limit"))
                continue
            if b"%PDF-" not in contents[:1024]:
                rejected.append(_reject(name, "invalid_pdf", "PDF header is missing"))
                continue

            content_hash = sha256(contents).hexdigest()
            document_id = _document_id(name, content_hash)
            if document_id in seen_ids:
                _LOG.info("Ignoring duplicate PDF upload: %s", name)
                continue
            seen_ids.add(document_id)

            try:
                reader = PdfReader(BytesIO(contents), strict=False)
                if reader.is_encrypted:
                    rejected.append(_reject(name, "encrypted_pdf", "Password-protected PDFs are unsupported"))
                    continue
                page_count = len(reader.pages)
            except Exception as exc:
                # pypdf may raise many parser-specific errors; never expose the
                # parser's raw exception content to the presentation layer.
                _LOG.warning("PDF reader rejected %s (%s)", name, type(exc).__name__)
                rejected.append(_reject(name, "invalid_pdf", "PDF could not be parsed"))
                continue

            if page_count == 0:
                rejected.append(_reject(name, "no_extractable_text", "PDF contains no pages"))
                continue
            if total_pages + page_count > ingestion.max_total_pages:
                raise DocumentValidationError("Upload exceeds aggregate PDF page limit")

            pages: list[DocumentPage] = []
            issues: list[ExtractionIssue] = []
            for page_number in range(1, page_count + 1):
                try:
                    extracted = reader.pages[page_number - 1].extract_text()
                    # pypdf may return None for a legitimate blank page.
                    if extracted is None:
                        extracted = ""
                    elif not isinstance(extracted, str):
                        raise TypeError("PDF text extraction returned a non-string value")
                except Exception as exc:
                    _LOG.warning(
                        "Text extraction failed for %s page %s (%s)",
                        name, page_number, type(exc).__name__,
                    )
                    extracted = ""
                    issues.append(ExtractionIssue(
                        page_number=page_number,
                        message="Text extraction failed for this page",
                    ))
                pages.append(DocumentPage(page_number=page_number, text=extracted))

            if not any(page.text.strip() for page in pages):
                fully_failed = len(issues) == page_count
                rejected.append(_reject(
                    name,
                    "extraction_failed" if fully_failed else "no_extractable_text",
                    (
                        "Text extraction failed on every PDF page"
                        if fully_failed else
                        "No selectable PDF text was found; scanned-only PDFs require OCR"
                    ),
                ))
                continue

            doc = RawDocument(
                document_id=document_id,
                file_name=name,
                content_sha256=content_hash,
                file_size_bytes=size,
                page_count=page_count,
                pages=tuple(pages),
                extraction_issues=tuple(issues),
            )

            # No partially chunked document is committed to the report.
            # A splitter failure is an operational error rather than a rejected PDF.
            doc_chunks: list[DocumentChunk] = []
            if splitter is None:
                splitter = _new_splitter(chunking)
            for page in doc.pages:
                if not page.text.strip():
                    continue
                try:
                    page_chunks = splitter.split_text(page.text)
                except Exception as exc:
                    raise ChunkingError(
                        f"Text splitting failed for {name} page {page.page_number}"
                    ) from exc
                if not isinstance(page_chunks, list) or any(
                    not isinstance(text, str) for text in page_chunks
                ):
                    raise ChunkingError(
                        f"Splitter returned invalid output for {name} page {page.page_number}"
                    )
                usable = [text for text in page_chunks if text.strip()]
                if not usable:
                    raise ChunkingError(
                        f"Nonempty text produced no chunks for {name} page {page.page_number}"
                    )
                for position, text in enumerate(usable):
                    if len(text) > chunking.chunk_size:
                        raise ChunkingError(
                            f"Chunk exceeds configured size for {name} page {page.page_number}"
                        )
                    if text not in page.text:
                        raise ChunkingError(
                            f"Chunk provenance cannot be verified for {name} page {page.page_number}"
                        )
                    doc_chunks.append(DocumentChunk(
                        chunk_id=_chunk_id(
                            document_id, page.page_number, position, text, chunking,
                        ),
                        document_id=document_id,
                        file_name=name,
                        page_number=page.page_number,
                        chunk_index=position,
                        text=text,
                    ))

            accepted.append(doc)
            chunks.extend(doc_chunks)
            total_pages += page_count

        return IngestionReport(
            raw_documents=tuple(accepted),
            chunks=tuple(chunks),
            rejected_files=tuple(rejected),
        )
