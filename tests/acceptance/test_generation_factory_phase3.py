"""Verify production composition with mocked LangChain/Google modules (offline)."""
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from pdf_rag.application.config import GenerationPolicy
from pdf_rag.domain.errors import ConfigurationError
from pdf_rag.domain.models import AnswerDraft
from pdf_rag.infrastructure.gemini.adapters import create_google_answer_generator


def test_missing_api_key_fails_without_google_import(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Prevent reading a local dotenv accidentally during an isolated unit test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    with pytest.raises(ConfigurationError):
        create_google_answer_generator()


def test_production_factory_constructs_structured_lcel_chain(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    fake_module = ModuleType("langchain_google_genai")
    fake_chat_instance = Mock()
    fake_structured = Mock()
    fake_chat_instance.with_structured_output.return_value = fake_structured
    created = []

    def fake_constructor(**kwargs):
        created.append(kwargs)
        return fake_chat_instance

    fake_module.ChatGoogleGenerativeAI = fake_constructor
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    # No real LangChain required: isolate factory wiring; the prompt's actual
    # ChatPromptTemplate is covered by the original prompt integration tests.
    from pdf_rag.infrastructure.gemini import prompt

    class PromptStub:
        def __or__(self, rhs):
            assert rhs is fake_structured
            return Mock(name="lcel_chain")

    monkeypatch.setattr(prompt, "build_qa_prompt", lambda: PromptStub())
    policy = GenerationPolicy(llm_model="gemini-3.8-flash", max_output_tokens=100)
    generator = create_google_answer_generator(policy=policy, api_key="local-test-key")
    assert created == [{
        "model": "gemini-3.8-flash", "temperature": 0.0,
        "max_output_tokens": 100, "google_api_key": "local-test-key",
    }]
    fake_chat_instance.with_structured_output.assert_called_once_with(
        AnswerDraft, method="json_schema", include_raw=True,
    )
    assert generator.bound_policy == policy
