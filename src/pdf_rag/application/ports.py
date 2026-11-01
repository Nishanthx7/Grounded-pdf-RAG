"""Step 3 infrastructure ports, unchanged."""
from dataclasses import dataclass
from typing import Protocol, Sequence
from pdf_rag.domain.models import GenerationResult, GroundingContext
from pdf_rag.application.config import GenerationPolicy

class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

@dataclass(frozen=True)
class VectorMatch:
    position: int
    similarity_score: float

class VectorSearchBackend(Protocol):
    def search(self, query_vector: Sequence[float], *, top_k: int) -> Sequence[VectorMatch]: ...

class VectorSearchBackendFactory(Protocol):
    def build(self, vectors: Sequence[Sequence[float]]) -> VectorSearchBackend: ...

class StructuredAnswerGenerator(Protocol):
    def generate(self, question: str, context: GroundingContext, *, policy: GenerationPolicy) -> GenerationResult: ...
