"""Run with: streamlit run app.py

Composition root; stores concrete services in the *current Streamlit session*.
No global cache and no disk-persisted PDF contents or FAISS indexes.
"""
from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from pdf_rag.application.config import GenerationPolicy, IndexBuildConfig
from pdf_rag.application.grounding import GroundingContextAssembler
from pdf_rag.application.services import DefaultRAGPipeline
from pdf_rag.domain.errors import RAGError
from pdf_rag.infrastructure.gemini.adapters import (
    create_google_answer_generator, create_google_embedding_provider,
)
from pdf_rag.infrastructure.pdf.processor import DefaultDocumentProcessor
from pdf_rag.infrastructure.vector.faiss_backend import FaissVectorSearchBackendFactory
from pdf_rag.infrastructure.vector.manager import SessionVectorStoreManager
from pdf_rag.presentation.streamlit_app import DefaultStreamlitApp, _safe_error

LOG = logging.getLogger(__name__)


def compose_app(ui=st) -> DefaultStreamlitApp | None:
    """Return a wired UI without creating new providers on every rerun."""
    load_dotenv()
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key or not key.strip() or key.strip() == "replace_with_your_gemini_api_key":
        ui.error("Missing Gemini API key. Create a .env file and set GOOGLE_API_KEY, then rerun the app.")
        return None
    # One manager, one pipeline and one Gemini provider pair per Streamlit session.
    if "_rag_services" not in ui.session_state:
        try:
            policy = GenerationPolicy()
            embeddings = create_google_embedding_provider(api_key=key)
            manager = SessionVectorStoreManager(embeddings, FaissVectorSearchBackendFactory())
            generator = create_google_answer_generator(api_key=key, policy=policy)
            pipeline = DefaultRAGPipeline(manager, generator, assembler=GroundingContextAssembler())
            index_config = IndexBuildConfig(
                embedding_model=embeddings.embedding_model,
                embedding_dimension=768,
                embedding_profile_id=embeddings.embedding_profile_id,
            )
            ui.session_state["_rag_services"] = (manager, pipeline, index_config, policy)
        except Exception as exc:
            LOG.warning("Failed to initialize RAG services (%s)", type(exc).__name__)
            ui.error(_safe_error(exc))
            return None
    manager, pipeline, index_config, policy = ui.session_state["_rag_services"]
    return DefaultStreamlitApp(
        DefaultDocumentProcessor(), manager, pipeline, ui=ui,
        index_config=index_config, generation=policy,
    )


if __name__ == "__main__":
    st.set_page_config(page_title="PDF Question Answering", page_icon="📄", layout="centered")
    app = compose_app(st)
    if app is not None:
        app.run()
