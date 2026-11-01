# Step 5 — Phase 1: DefaultDocumentProcessor

**Implemented:** PDF validation, page-wise pypdf extraction, deterministic chunking with LangChain's `RecursiveCharacterTextSplitter`, stable document/chunk IDs, deduplication, rejection reports, extraction warnings, and batch resource limits.

**Not implemented:** FAISS, Gemini, the RAG pipeline, and Streamlit. Their Step 4 RED tests and method stubs remain untouched.

## Architecture

- The implementation is in `src/pdf_rag/infrastructure/pdf/processor.py`.
- The original Step 4 test import (`from pdf_rag.application.services import DefaultDocumentProcessor`) still works through a compatibility re-export in `application/services.py`.
- The re-export is a temporary compatibility coupling across layers. When a composition root is introduced in a later phase, tests should import the concrete adapter from `infrastructure.pdf.processor`, leaving `application.services` dependent on protocols only.
- The existing Step 3 Pydantic schemas and service protocols are unchanged.
- The external `langchain-text-splitters` library is required at runtime. There is **no production fallback** that silently changes the chunking algorithm.

## Installation

From this folder, using Python 3.11+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-phase1.txt
python -m pip install -e . --no-deps
```

Then execute the phase-specific tests:

```bash
python -m pytest -q tests/acceptance/test_document_processor.py tests/acceptance/test_document_processor_boundaries.py
```

Run unchanged domain / PDF library tests:

```bash
python -m pytest -q tests/unit tests/integration/test_pdf_library.py
```

The remaining acceptance tests intentionally fail until their services are implemented in subsequent phases. Do not interpret that as a processor regression. `requirements-test.txt` includes additional FAISS-related dependencies for those later phases.

## Decision table

| Scenario | Behavior |
|---|---|
| Non-PDF extension or invalid header | Reject the file (`unsupported_file_type` / `invalid_pdf`) |
| Empty file or per-file byte limit | Reject the file (`empty_file` / `file_too_large`) |
| Encrypted PDF | Reject (`encrypted_pdf`) |
| Image-only/blank PDF | Reject (`no_extractable_text`) |
| PDF text extraction fails on every page | Reject (`extraction_failed`) |
| Individual page extraction fails but other pages contain text | Accept valid pages, record an `ExtractionIssue` for each failing page |
| Too many uploaded files, aggregate byte limit, aggregate page limit | Abort the batch with `DocumentValidationError` |
| Identical filename + bytes uploaded twice | Keep one copy |
| Same bytes with different filenames | Treat as two distinct documents |
| Chunk violates size or exact substring provenance | Abort with `ChunkingError`, do not return a partially indexed report |

**Page provenance:** Each `DocumentChunk` belongs to a single original PDF page, `page_number` starts at 1, and `chunk_index` starts at 0 within each page. The text is never merged across PDF pages. `chunk_id` includes a canonical encoding of parent identity, page, index, chunk text, and all chunking settings. This is stable across identical inputs and changes when chunking configuration changes.

**Privacy:** Extraction and chunking are fully local. Uploaded files are not persisted. Later Gemini indexing will transmit document text to Google and will need explicit appropriate user-facing disclosure. PDFs may contain malicious or pathological data; this implementation enforces byte and page limits, but it is not a parser sandbox or CPU/memory quota enforcement mechanism.

## Verification summary (current execution environment)

The target runtime here contains `pypdf` and Pydantic but **does not contain** `langchain-text-splitters`. Installing it failed due to network/DNS restrictions. Therefore the original LangChain integration could not be executed directly here.

- RED captured before implementation: original processor acceptance test raised `NotImplementedError`.
- GREEN after implementation: **13/13** new processor boundary tests passed using injected splitter test doubles (directly, no dependency substitute in package).
- GREEN after implementation: **27/27** processor-focused tests passed with an external, temporary **test-only** sliding-window stand-in placed on `PYTHONPATH`. That stand-in was not included in this project and does not verify the real LangChain split behavior.
- Baseline unchanged domain/PDF tests: **45/45** passed.

**Required follow-up verification:** Install the pinned real `langchain-text-splitters` distribution and rerun the 27 processor tests in your environment. Only after that may the processor be considered fully validated against the actual dependency.

## Explicit non-goals

No OCR, PDF decryption, file persistence, answer generation, FAISS storage, multi-user handling, or network calls. This module should not acquire those responsibilities in later phases.
