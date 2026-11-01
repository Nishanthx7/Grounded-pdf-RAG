# AI-Powered PDF Question Answering System

**Step 5, Phase 4 — Streamlit UI** (`pdf-rag-system 0.5.0`)

Complete source project including Phase 1 PDF ingestion, Phase 2 Gemini embeddings/FAISS, Phase 3 RAG grounding/citations, and Phase 4 Streamlit UI.

Start with **[PHASE4_README.md](PHASE4_README.md)** for Windows installation, tests, `.env` setup and `streamlit run app.py`. No API key is necessary to run offline unit/contract tests.

The working FAISS index is kept in Streamlit session memory only. The UI sends PDF text and questions to Google's Gemini API for processing. Do not upload confidential PDFs without reviewing the applicable terms.
