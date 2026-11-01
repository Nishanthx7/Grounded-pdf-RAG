"""RED: test UI session lifecycle without running Streamlit or calling external APIs.

Step 4 agrees that a changed upload selection automatically starts indexing;
unchanged uploaded bytes and filenames are not reindexed on subsequent reruns.
"""
from types import SimpleNamespace
import pytest
from pdf_rag.presentation.streamlit_app import DefaultStreamlitApp
from pdf_rag.domain.models import IngestionReport, PdfUpload
from tests._builders import chunk, chunks_for, document, index, valid_answer

pytestmark = pytest.mark.implementation


def rig(mocker):
    d = document(texts=("Evidence on page one",))
    c = chunks_for(d)
    processor = mocker.Mock()
    processor.process_uploads.return_value = IngestionReport(raw_documents=(d,), chunks=c)
    manager = mocker.Mock()
    manager.build_index.return_value = index(c)
    pipeline = mocker.Mock()
    pipeline.answer.return_value = valid_answer(c[0])
    ui = mocker.MagicMock()
    ui.session_state = {}
    ui.file_uploader.return_value = [SimpleNamespace(name="research.pdf", getvalue=lambda: b"%PDF test bytes")]
    ui.chat_input.return_value = None
    app = DefaultStreamlitApp(processor, manager, pipeline, ui=ui)
    return app, ui, processor, manager, pipeline


def test_repeated_streamlit_rerun_does_not_reembed_unchanged_uploads(mocker):
    app, ui, processor, manager, _ = rig(mocker)
    app.run()
    app.run()
    assert processor.process_uploads.call_count == 1
    assert manager.build_index.call_count == 1
    assert ui.session_state["vector_store"] is manager
    assert ui.session_state["vector_index"] == manager.build_index.return_value


def test_changed_upload_invalidates_old_chat_and_rebuilds_index(mocker):
    app, ui, processor, manager, _ = rig(mocker)
    app.run()
    ui.session_state["chat_history"] = ["old answer"]
    ui.file_uploader.return_value = [SimpleNamespace(name="new.pdf", getvalue=lambda: b"different PDF")]
    app.run()
    assert manager.build_index.call_count == 2
    assert ui.session_state["chat_history"] == []


def test_chat_question_uses_ready_index_without_reprocessing_pdf(mocker):
    app, ui, processor, manager, pipeline = rig(mocker)
    app.run()
    ui.chat_input.return_value = "What did the document say?"
    app.run()
    manager.build_index.assert_called_once()
    pipeline.answer.assert_called_once()
    assert pipeline.answer.call_args.args[1] == "What did the document say?"


def test_no_upload_never_builds_an_index(mocker):
    app, ui, processor, manager, pipeline = rig(mocker)
    ui.file_uploader.return_value = []
    app.run()
    manager.build_index.assert_not_called()
    pipeline.answer.assert_not_called()


def test_indexing_failure_never_publishes_partially_ready_index(mocker):
    from pdf_rag.domain.errors import EmbeddingError
    app, ui, processor, manager, _ = rig(mocker)
    manager.build_index.side_effect = EmbeddingError("mock quota issue")
    app.run()
    assert "vector_index" not in ui.session_state or ui.session_state["vector_index"] is None
    ui.error.assert_called()


def test_generation_quota_failure_does_not_append_fabricated_answer(mocker):
    from pdf_rag.domain.errors import QuotaExhaustedError
    app, ui, processor, manager, pipeline = rig(mocker)
    app.run()
    pipeline.answer.side_effect = QuotaExhaustedError("quota exhausted")
    ui.chat_input.return_value = "Question?"
    app.run()
    assert not any("The result is 42" in str(item) for item in ui.session_state.get("chat_history", []))
    ui.error.assert_called()
