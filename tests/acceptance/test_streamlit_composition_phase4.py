"""Offline composition-root checks. No network, Streamlit or Google account required."""
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from tests._builders import digest

pytestmark = pytest.mark.implementation
ROOT = Path(__file__).resolve().parents[2]


def load_entrypoint():
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.session_state = {}
    fake_streamlit.error = Mock()
    spec = importlib.util.spec_from_file_location("phase4_test_entrypoint", ROOT / "app.py")
    entry = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
        spec.loader.exec_module(entry)
    return entry, fake_streamlit


def test_compose_app_reuses_manager_and_generator_per_session():
    entry, ui = load_entrypoint()
    embeddings = Mock()
    embeddings.embedding_model = "gemini-embedding-2"
    embeddings.embedding_profile_id = digest("profile")
    generator = Mock()
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-only-non-secret"}, clear=True), \
         patch.object(entry, "load_dotenv"), \
         patch.object(entry, "create_google_embedding_provider", return_value=embeddings) as make_embed, \
         patch.object(entry, "create_google_answer_generator", return_value=generator) as make_gen:
        first = entry.compose_app(ui)
        second = entry.compose_app(ui)
    assert first is not None
    assert second is not None
    assert first.vector_manager is second.vector_manager
    assert first.pipeline is second.pipeline
    assert first.index_config.embedding_profile_id == embeddings.embedding_profile_id
    make_embed.assert_called_once()
    make_gen.assert_called_once()


def test_missing_gemini_key_returns_safe_visible_error():
    entry, ui = load_entrypoint()
    with patch.dict(os.environ, {}, clear=True), patch.object(entry, "load_dotenv"):
        assert entry.compose_app(ui) is None
    ui.error.assert_called_once()
    assert "GOOGLE_API_KEY" in ui.error.call_args.args[0]
    assert "_rag_services" not in ui.session_state


def test_failed_service_construction_does_not_publish_partial_session_services():
    entry, ui = load_entrypoint()
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake"}, clear=True), \
         patch.object(entry, "load_dotenv"), \
         patch.object(entry, "create_google_embedding_provider", side_effect=ValueError("my_secret")):
        assert entry.compose_app(ui) is None
    assert "_rag_services" not in ui.session_state
    assert "my_secret" not in ui.error.call_args.args[0]
