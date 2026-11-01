"""Gemini provider tests using the standard library Mock. Never use a live key."""
from unittest.mock import Mock, patch
import pytest

from pdf_rag.domain.errors import EmbeddingError, RateLimitError, QuotaExhaustedError
from pdf_rag.infrastructure.gemini.adapters import (
    GeminiEmbeddingProvider, embedding_profile_id, DOCUMENT_PREFIX, QUERY_PREFIX,
)

pytestmark = pytest.mark.implementation


def test_production_profile_formats_document_and_question_separately():
    client = Mock()
    client.embed_documents.return_value = [[1.0] * 768]
    client.embed_query.return_value = [2.0] * 768
    adapter = GeminiEmbeddingProvider(client, document_prefix=DOCUMENT_PREFIX,
                                      query_prefix=QUERY_PREFIX)
    assert len(adapter.embed_documents(["body"])[0]) == 768
    assert len(adapter.embed_query("What?")) == 768
    client.embed_documents.assert_called_once_with(
        ["title: none | text: body"], output_dimensionality=768)
    client.embed_query.assert_called_once_with(
        "task: question answering | query: What?", output_dimensionality=768)
    assert adapter.embedding_profile_id == embedding_profile_id()


def test_batch_limit_is_respected():
    client = Mock()
    client.embed_documents.side_effect = [[[1] * 768] * 2, [[1] * 768]]
    adapter = GeminiEmbeddingProvider(client, batch_size=2)
    assert len(adapter.embed_documents(["a", "b", "c"])) == 3
    assert client.embed_documents.call_count == 2
    assert client.embed_documents.call_args_list[0].args[0] == ["a", "b"]
    assert client.embed_documents.call_args_list[1].args[0] == ["c"]


def test_mismatched_response_length_rejected():
    client = Mock()
    client.embed_documents.return_value = [[1, 0]]
    with pytest.raises(EmbeddingError, match="count"):
        GeminiEmbeddingProvider(client).embed_documents(["alpha", "beta"])


def test_transient_rate_limit_retries_then_succeeds_without_sleeping():
    class TooManyRequests(Exception):
        status_code = 429
    client = Mock()
    client.embed_query.side_effect = [TooManyRequests("sensitive"), [1.0] * 768]
    adapter = GeminiEmbeddingProvider(client, max_attempts=2, backoff_seconds=0)
    assert len(adapter.embed_query("q")) == 768
    assert client.embed_query.call_count == 2


def test_exhausted_rate_limit_has_bounded_retries_and_safe_message():
    class TooManyRequests(Exception):
        status_code = 429
    client = Mock()
    client.embed_query.side_effect = TooManyRequests("secret-content")
    adapter = GeminiEmbeddingProvider(client, max_attempts=2, backoff_seconds=0)
    with pytest.raises(RateLimitError) as e:
        adapter.embed_query("q")
    assert "secret-content" not in str(e.value)
    assert client.embed_query.call_count == 2


def test_quota_exhausted_is_not_retried():
    QuotaExhausted = type("QuotaExhausted", (Exception,), {})
    client = Mock()
    client.embed_query.side_effect = QuotaExhausted("over quota")
    adapter = GeminiEmbeddingProvider(client, max_attempts=3, backoff_seconds=0)
    with pytest.raises(QuotaExhaustedError):
        adapter.embed_query("q")
    assert client.embed_query.call_count == 1


def test_blank_input_rejected_before_network():
    client = Mock()
    adapter = GeminiEmbeddingProvider(client)
    with pytest.raises(EmbeddingError):
        adapter.embed_query("  ")
    with pytest.raises(EmbeddingError):
        adapter.embed_documents(["valid", "  "])
    client.embed_query.assert_not_called()
    client.embed_documents.assert_not_called()


def test_missing_api_key_fails_before_importing_external_sdk(monkeypatch):
    from pdf_rag.domain.errors import ConfigurationError
    from pdf_rag.infrastructure.gemini.adapters import create_google_embedding_provider
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Disallow local .env reloading for a fully deterministic unit test.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
        create_google_embedding_provider()
