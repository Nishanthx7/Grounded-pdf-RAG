import pytest
from pydantic import ValidationError
from pdf_rag.application.config import (
    ChunkingPolicy, IngestionPolicy, IndexBuildConfig, RetrievalPolicy, GenerationPolicy
)
from tests._builders import digest


def test_locked_defaults():
    assert IngestionPolicy().max_files == 10
    assert IngestionPolicy().max_file_size_bytes == 20 * 1024 * 1024
    assert IngestionPolicy().max_total_pages == 300
    assert ChunkingPolicy().chunk_size == 1000
    assert ChunkingPolicy().chunk_overlap == 150
    assert RetrievalPolicy().top_k == 5
    assert RetrievalPolicy().similarity_threshold == .7
    assert IndexBuildConfig(embedding_profile_id=digest("profile")).embedding_dimension == 768

@pytest.mark.parametrize("policy", [
    lambda: ChunkingPolicy(chunk_size=100, chunk_overlap=100),
    lambda: ChunkingPolicy(chunk_size=100, chunk_overlap=101),
    lambda: ChunkingPolicy(separators=("\n", " ")),
    lambda: RetrievalPolicy(similarity_threshold=1.1),
    lambda: RetrievalPolicy(top_k=0),
    lambda: IngestionPolicy(max_files=0),
    lambda: GenerationPolicy(max_output_tokens=0),
    lambda: GenerationPolicy(context_token_budget=0),
    lambda: IndexBuildConfig(embedding_profile_id=digest("p"), embedding_dimension=512),
])
def test_invalid_policies_are_rejected(policy):
    with pytest.raises(ValidationError):
        policy()


def test_policies_frozen_and_strict():
    value = RetrievalPolicy()
    with pytest.raises(ValidationError):
        value.top_k = 7
    with pytest.raises(ValidationError):
        RetrievalPolicy(not_a_setting=42)
