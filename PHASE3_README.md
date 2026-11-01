# Step 5, Phase 3 — RAGPipeline and Gemini structured generation

**Version:** `pdf-rag-system 0.4.0` | **Scope:** application RAG pipeline and Gemini answer adapter **only**.

This archive is a **complete replacement working copy** of the Phase 2 project: it includes Phases 1–2 unchanged (except packaging metadata), and adds Phase 3. Do **not** manually merge it with the Phase 2 folder. Keep your Phase 2 folder as a backup.

## What was added

- `src/pdf_rag/application/grounding.py`: `GroundingContextAssembler` assigns `S1...` IDs to retrieved sources, attaches immutable filename/page/chunk provenance, and bounds context size. Without an exact provider tokenizer, V1 conservatively budgets **UTF-8 bytes** for the **retrieved context only** and leaves `context_token_count=None`; it **does not** misrepresent bytes as measured tokens. When a source is cut, the excerpt shows `[truncated]`. No fallback to irrelevant or made-up context.
- `src/pdf_rag/application/services.py`: `DefaultRAGPipeline` coordinates retrieval, refuses empty evidence, validates structured citation IDs against admitted evidence, checks `[S1]` references in `answer_text`, canonicalizes abstentions, retains provider token usage, and propagates typed failures.
- `src/pdf_rag/infrastructure/gemini/prompt.py`: an LCEL-compatible `ChatPromptTemplate`, with system policy kept separate from untrusted PDF excerpts, using only `question` and `context` inputs.
- `src/pdf_rag/infrastructure/gemini/adapters.py`: `GeminiStructuredAnswerGenerator`, optional `create_google_answer_generator()`, and structured Pydantic Gemini output via `with_structured_output(AnswerDraft, method="json_schema", include_raw=True)`. The raw `AIMessage` is used only to extract reported token usage. Configurable bounded retry with exponential backoff and jitter for transient provider errors. Live API use is **opt-in**.
- Phase 3 acceptance/unit-style tests cover citation manipulation, budget exhaustion, model configuration, 429/503 retries, 403 and daily quota errors, Pydantic response parsing, prompt boundary, and a cross-module manager→pipeline integration with fake vectors.

## Windows PowerShell setup

1. In Windows Explorer, right click `pdf_rag_step5_phase3.zip` -> **Extract All...** and choose `C:\Users\nishanth\Downloads\pdf_rag_step5_phase3`. The ZIP contains the project files at its **root** (no extra nested `pdf_rag_step5_phase3_project` directory).
2. Open a PowerShell terminal at the extracted project root. Verify that `pyproject.toml` and `requirements-phase3.txt` appear when you run `dir`.

```powershell
cd "C:\Users\nishanth\Downloads\pdf_rag_step5_phase3"
python -m pip install -r requirements-phase3.txt
python -m pip install -e . --no-deps
```

Run Phase 3 tests (no Gemini API key needed):

```powershell
python -m pytest -q tests/acceptance/test_rag_pipeline.py tests/acceptance/test_gemini_adapters.py tests/acceptance/test_rag_phase3_boundaries.py tests/acceptance/test_generation_adapter_phase3.py tests/acceptance/test_generation_factory_phase3.py tests/acceptance/test_rag_vector_integration_phase3.py tests/acceptance/test_prompt_contract.py tests/acceptance/test_lcel_chain.py
```

Then verify Phases 1–2 have not regressed:

```powershell
python -m pytest -q tests/acceptance/test_document_processor.py tests/acceptance/test_document_processor_boundaries.py
python -m pytest -q tests/acceptance/test_vector_manager.py tests/acceptance/test_vector_manager_phase2_boundaries.py tests/acceptance/test_embedding_adapter_phase2.py tests/acceptance/test_faiss_adapter_simulated.py tests/acceptance/test_faiss_adapter.py tests/integration/test_faiss_library.py
```

**Do not run `pytest` across every test module to judge Phase 3**: `tests/acceptance/test_streamlit_session.py` is still the intentionally RED Phase 4 work.

## Test verification performed in the development container

- **RED baseline**: first existing Phase 3 pipeline test failed with the expected `NotImplementedError`.
- **GREEN**: 52 Phase 3 deterministic tests passed (using a temporary, external-to-the-archive `mocker` fixture substitute because `pytest-mock` was not installed). This includes two cross-module vector/RAG tests; no live API requests.
- **Regression**: 76 non-implementation checks passed, with four optional library tests skipped; separately 35 Phase 2 acceptance/integration tests passed, with two native FAISS tests skipped. Counts overlap; do not add these figures together as unique tests.
- **Not verified in this container**: actual `langchain_core` `ChatPromptTemplate` and LCEL pipeline, native FAISS adapter, real LangChain splitting, live Gemini authentication/structured output. Network/PyPI were inaccessible here. Phases 1 and 2 previously passed with those libraries in the user's Windows environment (27 and 39 tests respectively). Your Windows tests are the authoritative integration gate.

## Configuration and safety

Use `.env.example` as a guide for adding `GOOGLE_API_KEY` only when **you choose** to test a live Gemini call. Never upload or commit `.env`. The offline acceptance tests do not use any credentials.

The default model is configurable through `GenerationPolicy` (`gemini-3.8-flash` at the time this was designed). The factory is imported and called explicitly; it is not run by the tests or on import. If changing LLM model, temperature, or max output tokens, create a new production generator for the new policy. Changing retrieval threshold does not re-embed PDFs.

**Limits:** schema-valid citations only prove the source was available to the model; they do not independently prove every sentence is supported. This requires additional semantic answer evaluation later. The context budget is only the evidence portion, not provider-reported full prompt tokens. Production uptime and live quota are not claimed.

## Scope explicitly excluded

`DefaultStreamlitApp` remains an unimplemented stub. No UI, chat history, authentication, vector persistence, OCR, or live model calls were added.
