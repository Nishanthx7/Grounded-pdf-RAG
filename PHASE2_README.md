# Step 5 — Phase 2: VectorStoreManager (FAISS + Gemini Embeddings)

**Scope:** This extends the Phase 1 archive; it does not replace or rewrite the existing document processor, Pydantic domain schemas, application protocols, RAG orchestration stubs, or Streamlit stubs.

## Implemented

- `SessionVectorStoreManager`: deterministic in-memory index descriptors, normalized 768-dimensional vectors, one vector per chunk, duplicate/empty/mismatched embedding rejection, safe errors, atomic index replacement, index reuse for unchanged inputs, clear/stale-index handling, and thresholded cosine retrieval.
- `FaissVectorSearchBackendFactory`: `faiss.IndexFlatIP` via contiguous `numpy.float32`; L2-normalizes vectors and queries to ensure scores are cosine similarities.
- `GeminiEmbeddingProvider`: injected LangChain embedding client, batch requests, explicit 768-dimensional output, query/document formatting for the **production factory**, deterministic embedding profile, safe error translation, bounded retry of transient rate-limit/server failures.
- `create_google_embedding_provider`: optional, explicit Google client construction using environment API key. No network calls occur when importing or testing the project.
- Test coverage: original vector-manager acceptance tests retained; more tests for invalid backends, fingerprint changes, cache reuse, failure atomicity, Gemini prefixes, batching, retries; FAISS boundary test with NumPy simulation. Original native FAISS tests retained for installed environments.

## Two important choices

1. **Deterministic tie-breaking**: The manager requests all index positions from the backend and chooses the globally sorted top `k` by `(-similarity_score, chunk_id)`. This is correct for V1's bounded single-user corpus, but it is O(number of chunks) per query in returned results. Revisit if corpus size grows.
2. **Google Embedding 2 task instructions**: The production provider formats chunks as `title: none | text: <chunk>` and questions as `task: question answering | query: <question>`, per Google documentation. The injected-client adapter's default uses no prefixes to preserve the frozen Step 4 injected-client acceptance contract. Never construct a *production* provider using the default no-prefix constructor; use the production factory.

Document embeddings are transmitted to Google **only if you deliberately instantiate the production Gemini provider**. Running these tests does not transmit PDFs or other text externally.

## Windows PowerShell setup

Extract ZIP; the ZIP contains a folder named `pdf_rag_step5_phase2_project`. Open PowerShell inside that folder. Then:

```powershell
python -m pip install -r requirements-phase2.txt
python -m pip install -e . --no-deps
```

### Verify Phase 1 has not regressed

```powershell
python -m pytest -q tests/acceptance/test_document_processor.py tests/acceptance/test_document_processor_boundaries.py
```

Expected based on previous user Windows verification: **27 passed**. They were not rerun with the real LangChain splitter in the handoff container, where that distribution is unavailable.

### Run the Phase 2 test suite (including native FAISS)

```powershell
python -m pytest -q tests/acceptance/test_vector_manager.py tests/acceptance/test_vector_manager_phase2_boundaries.py tests/acceptance/test_embedding_adapter_phase2.py tests/acceptance/test_faiss_adapter_simulated.py tests/acceptance/test_faiss_adapter.py tests/integration/test_faiss_library.py
```

Expected: all selected tests pass if FAISS is correctly installed; if FAISS cannot be imported, the native FAISS tests are skipped. **Do not count skipped native FAISS tests as validation.**

The original Step 4 injected-client Gemini embedding test needs `pytest-mock`. Run it separately (the other tests in that module target Phase 3's intentionally unimplemented answer generator):

```powershell
python -m pytest -q tests/acceptance/test_gemini_adapters.py -k embedding_adapter_calls_injected_client_not_network
```

Run the unchanged domain/PDF library checks:

```powershell
python -m pytest -q tests/unit tests/integration/test_pdf_library.py
```

### Local example (using an injected fake, no API call)

```python
from pdf_rag.application.services import SessionVectorStoreManager
from pdf_rag.infrastructure.vector.faiss_backend import FaissVectorSearchBackendFactory
from pdf_rag.application.config import IndexBuildConfig, RetrievalPolicy
from tests._builders import FakeEmbeddingProvider, chunk, config

provider = FakeEmbeddingProvider({"evidence": (1., 0.), "question": (1., 0.)})
manager = SessionVectorStoreManager(provider, FaissVectorSearchBackendFactory())
index = manager.build_index([chunk(text="evidence")], config=config())
result = manager.retrieve(index, "question", policy=RetrievalPolicy())
assert result.hits[0].chunk.text == "evidence"
```

### Real Gemini configuration (not executed as part of tests)

Provide a Google API key in `GOOGLE_API_KEY` or `GEMINI_API_KEY`, preferably through `.env` copied from `.env.example`; do not share the key in chat or commit it.

```python
from pdf_rag.infrastructure.gemini.adapters import create_google_embedding_provider
from pdf_rag.application.config import IndexBuildConfig

provider = create_google_embedding_provider()
config = IndexBuildConfig(
    embedding_model=provider.embedding_model,
    embedding_dimension=768,
    embedding_profile_id=provider.embedding_profile_id,
)
# Pass `provider` to SessionVectorStoreManager, with a FAISS backend factory.
```

No real Gemini API request was made in this handoff environment. An explicit live smoke test is still needed before calling the Google integration end-to-end verified.

## Validation record

- Original Step 4 vector manager RED observed: `NotImplementedError` on `build_index`.
- Initial manager GREEN: 13/14 passed; one failed because the **Step 4 test fixture incorrectly assigned a different `document_id` per text chunk of the same source filename**. The test builder was repaired so same-file chunks share a parent document identity, preserving the domain invariant that `document_count` counts distinct `document_id` values. No original test assertions were changed.
- After repair and implementation: original vector manager acceptance tests **14/14 passed**.
- Expanded tests plus unaffected domain/PDF baseline: **80/80 passed locally**, using a NumPy FAISS simulation, not native FAISS.
- Native FAISS tests: **skipped locally** because `faiss-cpu` could not be installed in the network-isolated container.
- Original Phase 1 tests: previously **27/27 passed in the user's Windows environment** with real `langchain-text-splitters`; not rerun in this container.
- The original Step 4 `pytest-mock` test uses a dependency not available in the isolated container; rerun on Windows after dependencies are installed.

## Still deliberately RED (Phase 3/4)

`DefaultRAGPipeline.answer`, `GeminiStructuredAnswerGenerator.generate`, structured prompting, citation validation, and Streamlit session UI are unchanged/unimplemented. Do not run the entire `pytest -m implementation` suite expecting all tests to pass yet.

## Security and operational constraints

This is one in-memory index per manager/session. There is no FAISS serialization, multitenancy, encrypted storage, or durable persistence. API keys are never placed in fingerprints or log messages. Free-tier availability is subject to Google's current quotas and data-handling terms. Invalid provider responses and stale handles are rejected.
