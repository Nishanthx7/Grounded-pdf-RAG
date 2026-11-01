"""Deterministic structured generation adapter tests (offline; no LangChain import)."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pdf_rag.application.config import GenerationPolicy
from pdf_rag.domain.errors import (
    AnswerGenerationError, AnswerValidationError, ConfigurationError,
    QuotaExhaustedError, RateLimitError,
)
from pdf_rag.domain.models import AnswerDraft, AnswerStatus
from pdf_rag.infrastructure.gemini.adapters import GeminiStructuredAnswerGenerator
from tests._builders import chunk, grounded


CONTEXT = grounded((chunk(text="Result 42"),))
DRAFT = AnswerDraft(answer_text="Result 42 [S1]", status=AnswerStatus.ANSWERED, citation_ids=("S1",))


def generate(chain, *, policy=None, **kwargs):
    adapter = GeminiStructuredAnswerGenerator(chain, **kwargs)
    return adapter.generate("What result?", CONTEXT, policy=policy or GenerationPolicy())


def test_structured_raw_response_preserves_tokens():
    chain = Mock()
    chain.invoke.return_value = {
        "parsed": DRAFT, "raw": SimpleNamespace(usage_metadata={"input_tokens": 25, "output_tokens": 11, "total_tokens": 36}),
        "parsing_error": None,
    }
    response = generate(chain)
    assert response.draft == DRAFT
    assert response.token_usage.total_tokens == 36
    assert response.token_usage.input_tokens == 25


def test_usage_missing_is_unknown_not_zero():
    chain = Mock()
    chain.invoke.return_value = {"parsed": DRAFT, "raw": SimpleNamespace(usage_metadata=None), "parsing_error": None}
    assert generate(chain).token_usage.total_tokens is None


@pytest.mark.parametrize("value", [
    {"raw": None, "parsed": None, "parsing_error": ValueError("broken JSON")},
    {"raw": None, "parsed": {"status": "answered", "answer_text": "hello", "citation_ids": ["INVALID"]}, "parsing_error": None},
    "not a valid structured response",
])
def test_malformed_structured_response_is_not_a_success(value):
    chain = Mock()
    chain.invoke.return_value = value
    with pytest.raises(AnswerValidationError):
        generate(chain)


class ProviderError(Exception):
    def __init__(self, status_code, message="request failed"):
        self.status_code = status_code
        super().__init__(message)


def test_429_retry_is_bounded_and_eventually_succeeds():
    chain = Mock()
    chain.invoke.side_effect = [ProviderError(429), ProviderError(503), DRAFT]
    sleeps = []
    response = generate(chain, max_attempts=3, backoff_seconds=0.1, sleep=sleeps.append, jitter=lambda: 0.0)
    assert response.draft == DRAFT
    assert chain.invoke.call_count == 3
    assert sleeps == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize("status,expected", [(429, RateLimitError), (503, AnswerGenerationError)])
def test_transient_error_exhaustion_is_typed(status, expected):
    chain = Mock()
    chain.invoke.side_effect = ProviderError(status)
    with pytest.raises(expected):
        generate(chain, max_attempts=3, backoff_seconds=0, sleep=lambda _: None)
    assert chain.invoke.call_count == 3


def test_daily_quota_exhaustion_does_not_retry():
    chain = Mock()
    chain.invoke.side_effect = ProviderError(429, "daily quota exceeded")
    with pytest.raises(QuotaExhaustedError):
        generate(chain, max_attempts=3, sleep=lambda _: None)
    chain.invoke.assert_called_once()


def test_403_configuration_failure_does_not_retry():
    chain = Mock()
    chain.invoke.side_effect = ProviderError(403)
    with pytest.raises(ConfigurationError):
        generate(chain, max_attempts=3, sleep=lambda _: None)
    chain.invoke.assert_called_once()


def test_changed_model_policy_does_not_silently_reuse_old_chain():
    chain = Mock()
    chain.invoke.return_value = DRAFT
    adapter = GeminiStructuredAnswerGenerator(chain, bound_policy=GenerationPolicy(llm_model="gemini-3.8-flash"))
    with pytest.raises(ConfigurationError):
        adapter.generate("Question", CONTEXT, policy=GenerationPolicy(llm_model="other-model"))
    chain.invoke.assert_not_called()


def test_unsupported_prompt_version_rejected_before_invoke():
    chain = Mock()
    with pytest.raises(ConfigurationError):
        generate(chain, policy=GenerationPolicy(prompt_version="v2"))
    chain.invoke.assert_not_called()


def test_unexpected_api_failure_does_not_leak_provider_message():
    chain = Mock()
    chain.invoke.side_effect = RuntimeError("API key secret x: contents of confidential PDF")
    with pytest.raises(AnswerGenerationError) as captured:
        generate(chain)
    assert "secret" not in str(captured.value)
    assert "confidential" not in str(captured.value)


def test_draft_as_mapping_is_validated_by_pydantic():
    chain = Mock()
    chain.invoke.return_value = DRAFT.model_dump()
    response = generate(chain)
    assert response.draft == DRAFT
