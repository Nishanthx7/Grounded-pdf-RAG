"""Static protocol structural acceptance; no service behavior is claimed here."""
import inspect
from pdf_rag.application.contracts import DocumentProcessor, VectorStoreManager, RAGPipeline, StreamlitApp
from pdf_rag.application.ports import EmbeddingProvider, VectorSearchBackendFactory, StructuredAnswerGenerator
from pdf_rag.application.services import DefaultDocumentProcessor, SessionVectorStoreManager, DefaultRAGPipeline
from pdf_rag.presentation.streamlit_app import DefaultStreamlitApp
from tests._builders import FakeEmbeddingProvider, FakeVectorFactory, FakeGenerator


def test_core_public_methods_match_step3_protocol_names():
    expected = {
        DocumentProcessor: {"process_uploads"},
        VectorStoreManager: {"build_index", "retrieve", "clear"},
        RAGPipeline: {"answer"},
        StreamlitApp: {"run"},
    }
    for interface, names in expected.items():
        assert set(interface.__dict__) & names == names


def test_concrete_seams_expose_public_protocol_methods():
    assert hasattr(DefaultDocumentProcessor, "process_uploads")
    assert {"build_index", "retrieve", "clear"} <= set(dir(SessionVectorStoreManager))
    assert hasattr(DefaultRAGPipeline, "answer")
    assert hasattr(DefaultStreamlitApp, "run")


def test_fakes_expose_provider_port_capabilities():
    assert {"embed_documents", "embed_query"} <= set(dir(FakeEmbeddingProvider))
    assert "build" in dir(FakeVectorFactory)
    assert "generate" in dir(FakeGenerator)


def test_service_method_signatures_have_keyword_only_policies():
    for cls, method, policy in [
        (DefaultDocumentProcessor, "process_uploads", "ingestion"),
        (SessionVectorStoreManager, "build_index", "config"),
        (SessionVectorStoreManager, "retrieve", "policy"),
        (DefaultRAGPipeline, "answer", "generation"),
    ]:
        parameter = inspect.signature(getattr(cls, method)).parameters[policy]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
