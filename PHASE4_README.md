# Phase 4 — Streamlit UI (Step 5)

**Package**: `pdf-rag-system 0.5.0` | **Scope**: Streamlit presentation and session composition only.

This ZIP contains the entire updated project, including Phases 1–3 and their tests. Do not merge or overwrite earlier phase folders manually. Phase 3 remains a backup. **The ZIP stores project files directly at its root**.

## Windows setup (PowerShell)

Extract `pdf_rag_step5_phase4.zip` into:

```text
C:\Users\nishanth\Downloads\pdf_rag_step5_phase4
```

Open PowerShell in that folder, where `app.py`, `pyproject.toml`, and `requirements-phase4.txt` appear:

```powershell
cd "C:\Users\nishanth\Downloads\pdf_rag_step5_phase4"
python -m pip install -r requirements-phase4.txt
python -m pip install -e . --no-deps
```

Confirm Python 3.11+ and installed Streamlit:

```powershell
python -m streamlit --version
```

### Run the offline Phase 4 tests (no API key required)

```powershell
python -m pytest -q tests/acceptance/test_streamlit_session.py tests/acceptance/test_streamlit_phase4_boundaries.py tests/acceptance/test_streamlit_composition_phase4.py
```

Expected based on development-environment execution: **23 passed**. Then run the complete regression suite on Windows (with all dependencies installed):

```powershell
python -m pytest -q
```

If any tests fail, share their tracebacks; do not assume the Phase 4 package fixes failures in older code. The new Streamlit code is tested offline through a mocked UI; this environment did not have native Streamlit installed to launch the interactive browser.

### Launch interactive app

Make a `.env` file in **this project root** (never commit or share it):

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace the placeholder with your valid Gemini API key (not `replace_with_your_gemini_api_key`). The key remains in your local `.env` and is not shown in the app.

```powershell
python -m streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. Upload one or more selectable-text PDFs, confirm the index is ready, and ask an independent question. Rejected files and extraction warnings appear in the UI. Source expanders show the supporting PDF filename, page, chunk ID and excerpt.

**Privacy**: PDF text and questions are sent to Google's Gemini service for embedding/generation. This is NOT offline-only or suitable for confidential files without assessing Google's data handling. V1 uses session-only memory; refreshing/reconnecting may clear it.

## Implemented behavior

- Multi-PDF uploads with original `PdfUpload` validation and 10-file, 20-MB/file, 100-MB aggregate, 300-page limits; Streamlit config caps each uploaded file at 20 MB.
- Indexing status and milestone progress; never publishes partial indexes. Failed attempts do not silently retry on ordinary reruns.
- Upload content fingerprint includes original bytes, filename, ingestion/chunking and embedding config; changes invalidate existing vectors/history; uploader reordering does not re-embed.
- One manager, FAISS index, Gemini embedding provider, and answer generator per Streamlit session via composition root in `app.py`.
- Session-only chat transcript; previous chat turns are never submitted to `RAGPipeline.answer()` as memory.
- Expandable source citations with PDF filename, original page, chunk ID and exact chunk excerpt.
- Reset files/chat and manually retry indexing; source excerpts and history cleared when corpus changes.
- Safe error messages for quota exhaustion, rate limits, missing API keys, indexing failures and fabricated citation failures. Raw exception contents/secrets are not displayed.

## Verification performed in the development container

1. **RED:** the original six Streamlit session tests initially failed at the `NotImplementedError`.
2. **GREEN:** all **23** Phase 4 presentation and composition tests passed using deterministic UI/service mocks (temporary minimal `mocker` fixture because pytest-mock unavailable here).
3. **Regression:** **128** targeted domain/vector/RAG checks passed (no Streamlit module involved).
4. The full suite with native integrations unavailable in this container yielded 7 older PDF processing test failures (missing `langchain-text-splitters`), 4 skipped tests (missing FAISS/LangChain Core), and 158 other passes **before** the additional 17 Phase 4 tests were added. These missing dependencies do not indicate a known change to Phase 1, and Phase 1 previously passed on the user's Windows system.
5. **Not yet verified**: actual Streamlit browser interaction, live Gemini authentication, real API quotas, and end-to-end quality on diverse PDFs. Run local tests and manual smoke test on Windows before accepting Phase 4.

## What has not changed

Domain models, Step 3 service protocols, PDF extraction, FAISS manager, RAG grounding, prompt assembly and Gemini adapters are unchanged. V1 remains single-user/local and text-PDF-only. No OCR, permanent index serialization, authentication, semantic reranking or conversational memory was added.

## Suggested manual smoke checks

1. Upload `tests/fixtures/one_page.pdf`; confirm a ready index and sources.
2. Ask an answerable question then an unrelated question; check citations/abstention.
3. Type another question: no reindexing should occur.
4. Replace uploaded PDFs: chat and sources should reset.
5. Reset the workspace: file picker should empty and answer input disable.
6. Disconnect API key or exceed a known quota: UI should display an actionable error, without an invented assistant answer.

A live Gemini call is subject to quota and model availability; the offline suite does not prove it will succeed.
