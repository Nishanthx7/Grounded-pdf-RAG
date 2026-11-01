"""RED: actual LCEL prompt composition with a local fake chat model; no Gemini calls."""
import pytest
pytest.importorskip("langchain_core")
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from pdf_rag.infrastructure.gemini.prompt import build_qa_prompt
pytestmark = pytest.mark.implementation


def test_lcel_prompt_chain_passes_system_policy_and_pdf_evidence_as_separate_messages():
    captured = []
    def fake_chat(prompt_value):
        messages = prompt_value.to_messages()
        captured.extend(messages)
        return "MOCKED ANSWER"
    chain = build_qa_prompt() | RunnableLambda(fake_chat)
    response = chain.invoke({"question": "What does it say?", "context": "[S1] Evidence text."})
    assert response == "MOCKED ANSWER"
    assert any(isinstance(m, SystemMessage) for m in captured)
    assert any(isinstance(m, HumanMessage) and "[S1] Evidence text." in str(m.content) for m in captured)
    assert all("[S1] Evidence text." not in str(m.content) for m in captured if isinstance(m, SystemMessage))
