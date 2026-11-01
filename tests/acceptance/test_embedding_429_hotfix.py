"""Regression tests for Phase 4 rate limiting; no live Gemini calls."""
from unittest.mock import Mock
import pytest

from pdf_rag.domain.errors import RateLimitError, QuotaExhaustedError
from pdf_rag.infrastructure.gemini.adapters import GeminiEmbeddingProvider


def test_paces_only_between_document_batches():
    client = Mock()
    client.embed_documents.side_effect = [[[0.0] * 768] * 2, [[0.0] * 768]]
    delays: list[float] = []
    provider = GeminiEmbeddingProvider(client, batch_size=2, batch_interval_seconds=25,
                                       rate_limit_backoff_seconds=30, sleep=delays.append)
    assert len(provider.embed_documents(["a", "b", "c"])) == 3
    assert delays == [25]
    assert client.embed_documents.call_count == 2


def test_nested_429_backoff_then_recovers_without_repeating_previous_batches():
    class ResourceExhausted(Exception):
        status_code = 429
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = ResourceExhausted("hidden request")
    client = Mock()
    client.embed_documents.side_effect = [
        [[0.0] * 768] * 2,
        wrapped,
        [[0.0] * 768] * 2,
    ]
    delays: list[float] = []
    provider = GeminiEmbeddingProvider(
        client, batch_size=2, max_attempts=2, backoff_seconds=0,
        rate_limit_backoff_seconds=30, batch_interval_seconds=25,
        sleep=delays.append,
    )
    assert len(provider.embed_documents(["a", "b", "c", "d"])) == 4
    assert delays == [25, 30]
    assert client.embed_documents.call_count == 3


def test_nested_429_exhaustion_is_safely_classified():
    class ResourceExhausted(Exception):
        status_code = 429
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = ResourceExhausted("private client data")
    client = Mock()
    client.embed_query.side_effect = wrapped
    provider = GeminiEmbeddingProvider(client, max_attempts=2,
                                       backoff_seconds=0, rate_limit_backoff_seconds=0,
                                       sleep=lambda _: None)
    with pytest.raises(RateLimitError) as info:
        provider.embed_query("q")
    assert "private client data" not in str(info.value)
    assert client.embed_query.call_count == 2


def test_detects_daily_limit_and_does_not_retry():
    class ResourceExhausted(Exception):
        status_code = 429
    client = Mock()
    client.embed_query.side_effect = ResourceExhausted("requests per day limit reached")
    provider = GeminiEmbeddingProvider(client, max_attempts=3,
                                       backoff_seconds=0, rate_limit_backoff_seconds=0,
                                       sleep=lambda _: None)
    with pytest.raises(QuotaExhaustedError):
        provider.embed_query("q")
    client.embed_query.assert_called_once()


def test_query_embedding_does_not_incur_batch_pacing():
    client = Mock()
    client.embed_query.return_value = [0.0] * 768
    delays: list[float] = []
    provider = GeminiEmbeddingProvider(client, batch_interval_seconds=25,
                                       sleep=delays.append)
    provider.embed_query("What are ML steps?")
    assert delays == []


def test_four_batch_simulation_paces_three_times():
    client = Mock()
    client.embed_documents.side_effect = [
        [[0.0] * 768] * 32, [[0.0] * 768] * 32,
        [[0.0] * 768] * 32, [[0.0] * 768] * 18,
    ]
    delays: list[float] = []
    provider = GeminiEmbeddingProvider(client, batch_size=32,
                                       batch_interval_seconds=25, sleep=delays.append)
    assert len(provider.embed_documents(["chunk"] * 114)) == 114
    assert delays == [25, 25, 25]


def test_non_transient_error_does_not_retry():
    class InvalidArgument(Exception):
        status_code = 400
    client = Mock()
    client.embed_query.side_effect = InvalidArgument("private text")
    provider = GeminiEmbeddingProvider(client, max_attempts=3,
                                       sleep=lambda _: None)
    with pytest.raises(Exception) as info:
        provider.embed_query("q")
    assert "private text" not in str(info.value)
    client.embed_query.assert_called_once()
