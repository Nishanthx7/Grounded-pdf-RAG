"""Phase 4 presentation contracts: all fakes, no Streamlit server or Gemini calls."""
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from pdf_rag.application.config import ChunkingPolicy, RetrievalPolicy
from pdf_rag.domain.errors import QuotaExhaustedError
from pdf_rag.domain.models import IngestionReport
from pdf_rag.presentation.streamlit_app import DefaultStreamlitApp, _safe_error
from tests._builders import chunks_for, document, index, valid_answer

pytestmark = pytest.mark.implementation


def setup_app():
    doc = document(texts=("The answer is in the PDF.",))
    chunks = chunks_for(doc)
    processor = Mock()
    processor.process_uploads.return_value = IngestionReport(raw_documents=(doc,), chunks=chunks)
    manager = Mock()
    manager.build_index.return_value = index(chunks)
    pipeline = Mock()
    pipeline.answer.return_value = valid_answer(chunks[0])
    ui = MagicMock()
    ui.session_state = {}
    ui.sidebar.button.return_value = False
    ui.file_uploader.return_value = [SimpleNamespace(name="report.pdf", getvalue=lambda: b"%PDF-test data")]
    ui.chat_input.return_value = None
    app = DefaultStreamlitApp(processor, manager, pipeline, ui=ui)
    return app, ui, processor, manager, pipeline, chunks


def test_failures_do_not_repeat_api_calls_on_each_rerun():
    app, ui, processor, manager, _, _ = setup_app()
    manager.build_index.side_effect = QuotaExhaustedError("API key should not appear here")
    app.run()
    app.run()
    assert manager.build_index.call_count == 1
    assert processor.process_uploads.call_count == 1
    assert ui.session_state["index_status"] == "failed"
    assert ui.session_state["vector_index"] is None
    assert "API key should not appear here" not in ui.session_state["index_error"]


def test_retry_action_retries_once_without_new_upload():
    app, ui, _, manager, _, _ = setup_app()
    manager.build_index.side_effect = [QuotaExhaustedError("limit"), manager.build_index.return_value]
    app.run()
    ui.sidebar.button.side_effect = [False, True]  # reset, retry
    app.run()
    assert manager.build_index.call_count == 2
    assert ui.session_state["index_status"] == "ready"


def test_removed_files_invalidate_index_and_history():
    app, ui, _, manager, _, _ = setup_app()
    app.run()
    ui.session_state["chat_history"] = [{"role": "user", "content": "old"}]
    ui.file_uploader.return_value = []
    app.run()
    assert ui.session_state["vector_index"] is None
    assert ui.session_state["chat_history"] == []
    assert ui.session_state["index_status"] == "empty"
    assert manager.clear.call_count >= 1


def test_upload_order_change_does_not_reindex():
    app, ui, processor, manager, _, _ = setup_app()
    a = SimpleNamespace(name="a.pdf", getvalue=lambda: b"a")
    b = SimpleNamespace(name="b.pdf", getvalue=lambda: b"b")
    ui.file_uploader.return_value = [a, b]
    app.run()
    ui.file_uploader.return_value = [b, a]
    app.run()
    manager.build_index.assert_called_once()
    processor.process_uploads.assert_called_once()


def test_config_change_invalidates_index():
    app, ui, _, manager, _, _ = setup_app()
    app.run()
    app.chunking = ChunkingPolicy(chunk_size=600, chunk_overlap=100)
    app.run()
    assert manager.build_index.call_count == 2


def test_generation_error_has_no_assistant_message():
    app, ui, _, _, pipeline, _ = setup_app()
    app.run()
    ui.chat_input.return_value = "Question?"
    pipeline.answer.side_effect = QuotaExhaustedError("private content")
    app.run()
    assert [msg["role"] for msg in ui.session_state["chat_history"]] == ["user"]
    assert "private content" not in str(ui.session_state)
    ui.error.assert_called()


def test_chat_replay_is_not_passed_to_pipeline_and_sources_expand():
    app, ui, _, _, pipeline, chunks = setup_app()
    app.run()
    ui.chat_input.return_value = "What was the result?"
    app.run()
    assert len(ui.session_state["chat_history"]) == 2
    assert ui.session_state["chat_history"][1]["sources"][0]["page_number"] == 1
    assert ui.session_state["chat_history"][1]["sources"][0]["chunk_id"] == chunks[0].chunk_id
    ui.text.assert_called_with(chunks[0].text)
    ui.chat_input.return_value = "Different question"
    app.run()
    assert pipeline.answer.call_count == 2
    assert pipeline.answer.call_args.args[1] == "Different question"
    assert len(pipeline.answer.call_args.args) == 2


def test_reset_rotates_widget_key_and_clears_transcript():
    app, ui, _, _, _, _ = setup_app()
    app.run()
    ui.file_uploader.return_value = []  # Real new widget key has no files
    ui.sidebar.button.side_effect = [True, False]
    ui.session_state["chat_history"] = [{"role": "user", "content": "old"}]
    app.run()
    assert ui.session_state["upload_widget_version"] == 1
    assert ui.file_uploader.call_args.kwargs["key"] == "pdf_upload_1"
    assert ui.session_state["chat_history"] == []


def test_empty_pdf_corpus_does_not_attempt_embeddings():
    app, ui, processor, manager, pipeline, _ = setup_app()
    processor.process_uploads.return_value = IngestionReport()
    app.run()
    assert ui.session_state["index_status"] == "failed"
    manager.build_index.assert_not_called()
    pipeline.answer.assert_not_called()


def test_empty_question_is_ignored():
    app, ui, _, _, pipeline, _ = setup_app()
    app.run()
    ui.chat_input.return_value = "  "
    app.run()
    pipeline.answer.assert_not_called()


def test_missing_upload_does_not_call_pipeline_even_with_question():
    app, ui, _, manager, pipeline, _ = setup_app()
    ui.file_uploader.return_value = []
    ui.chat_input.return_value = "Question?"
    app.run()
    manager.build_index.assert_not_called()
    pipeline.answer.assert_not_called()


def test_error_type_translation_never_leaks_raw_text():
    assert "top_secret_token" not in _safe_error(QuotaExhaustedError("top_secret_token"))
    assert "untrusted" not in _safe_error(ValueError("untrusted"))


def test_indexing_progress_steps_and_readiness():
    app, ui, _, manager, _, _ = setup_app()
    app.run()
    progress = ui.progress.return_value
    assert progress.progress.call_args_list[0].args[0] == 45
    assert progress.progress.call_args_list[-1].args[0] == 100
    progress.empty.assert_called_once()
    assert ui.session_state["vector_store"] is manager


def test_invalid_upload_metadata_never_leaves_old_index_ready():
    app, ui, _, manager, pipeline, _ = setup_app()
    app.run()
    ui.file_uploader.return_value = [SimpleNamespace(name="", getvalue=lambda: b"123")]
    ui.chat_input.return_value = "Could be stale?"
    app.run()
    assert ui.session_state["vector_index"] is None
    assert ui.session_state["index_status"] == "failed"
    pipeline.answer.assert_not_called()
    manager.clear.assert_called()