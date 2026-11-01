"""Step 3 policy contracts, unchanged for Step 4."""
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pdf_rag.domain.models import Sha256

class PolicyModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class IngestionPolicy(PolicyModel):
    max_files: int = Field(default=10, gt=0)
    max_file_size_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_total_size_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_total_pages: int = Field(default=300, gt=0)

class ChunkingPolicy(PolicyModel):
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    separators: tuple[str, ...] = ("\n\n", "\n", " ", "")

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        if not self.separators or self.separators[-1] != "":
            raise ValueError("Final separator must be empty")
        return self

class IndexBuildConfig(PolicyModel):
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: Literal[768] = 768
    embedding_profile_id: Sha256
    similarity_metric: Literal["cosine"] = "cosine"
    normalization_enabled: Literal[True] = True

class RetrievalPolicy(PolicyModel):
    top_k: int = Field(default=5, gt=0)
    similarity_threshold: float = Field(default=0.70, ge=-1.0, le=1.0)

class GenerationPolicy(PolicyModel):
    llm_model: str = "gemini-3.8-flash"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    context_token_budget: int = Field(default=4096, gt=0)
    prompt_version: str = Field(default="v1", min_length=1)
