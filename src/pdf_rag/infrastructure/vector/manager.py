"""Session-scoped vector index application service.

This service owns a single *ready* vector index, never an in-progress index.
Its dependencies are application ports, not FAISS or Gemini concrete clients.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Sequence

from pdf_rag.application.config import IndexBuildConfig, RetrievalPolicy
from pdf_rag.application.ports import EmbeddingProvider, VectorSearchBackend, VectorSearchBackendFactory
from pdf_rag.domain.errors import EmbeddingError, IndexBuildError, IndexNotFoundError, RetrievalError
from pdf_rag.domain.models import DocumentChunk, RetrievalHit, RetrieverResult, VectorIndex

LOG = logging.getLogger(__name__)


def _unit_vector(values: Sequence[float], dimension: int) -> list[float]:
    """Reject invalid embeddings before they can reach a vector backend."""
    try:
        numbers = [float(item) for item in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError("Embedding contains nonnumeric values") from exc
    if len(numbers) != dimension:
        raise EmbeddingError(f"Expected embedding dimension {dimension}; received {len(numbers)}")
    if any(not math.isfinite(value) for value in numbers):
        raise EmbeddingError("Embedding contains nonfinite values")
    magnitude = math.hypot(*numbers)
    if magnitude == 0 or not math.isfinite(magnitude):
        raise EmbeddingError("Embedding has zero or unrepresentable norm")
    return [value / magnitude for value in numbers]


def _index_fingerprint(chunks: Sequence[DocumentChunk], config: IndexBuildConfig) -> str:
    payload = {
        "v": 1,
        # Include source metadata/text as defense against accidentally reused or
        # forged chunk IDs. Legitimate processor IDs already bind these fields.
        "chunks": [c.model_dump() for c in chunks],
        "embedding_model": config.embedding_model,
        "embedding_dimension": config.embedding_dimension,
        "embedding_profile_id": config.embedding_profile_id,
        "similarity_metric": config.similarity_metric,
        "normalization_enabled": config.normalization_enabled,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SessionVectorStoreManager:
    """Atomically builds/replaces a single in-memory index and resolves FAISS hits."""

    def __init__(self, embedding_provider: EmbeddingProvider, backend_factory: VectorSearchBackendFactory) -> None:
        self._embedding_provider = embedding_provider
        self._backend_factory = backend_factory
        self._index: VectorIndex | None = None
        self._chunks: tuple[DocumentChunk, ...] = ()
        self._backend: VectorSearchBackend | None = None

    def build_index(self, chunks: Sequence[DocumentChunk], *, config: IndexBuildConfig) -> VectorIndex:
        snapshot = tuple(chunks)
        if not snapshot:
            raise IndexBuildError("Cannot build an index with no document chunks")
        identities = [c.chunk_id for c in snapshot]
        if len(set(identities)) != len(identities):
            raise IndexBuildError("Duplicate document chunk IDs")
        if any(not c.text.strip() for c in snapshot):
            raise IndexBuildError("Cannot index empty chunks")
        provider_profile = getattr(self._embedding_provider, "embedding_profile_id", None)
        if provider_profile is not None and provider_profile != config.embedding_profile_id:
            raise IndexBuildError("Embedding configuration does not match provider profile")
        provider_model = getattr(self._embedding_provider, "embedding_model", None)
        if provider_model is not None and provider_model != config.embedding_model:
            raise IndexBuildError("Embedding model does not match provider configuration")

        fingerprint = _index_fingerprint(snapshot, config)
        if self._index is not None and self._index.index_id == fingerprint:
            return self._index  # no document re-embedding for a matching corpus/config

        try:
            result = self._embedding_provider.embed_documents([c.text for c in snapshot])
        except EmbeddingError:
            raise
        except Exception as exc:
            LOG.warning("Document embedding provider failed (%s)", type(exc).__name__)
            raise EmbeddingError("Document embedding request failed") from exc

        try:
            vectors = list(result)
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding provider returned an invalid collection") from exc
        if len(vectors) != len(snapshot):
            raise EmbeddingError("Embedding count does not match document chunk count")
        normalized = [_unit_vector(v, config.embedding_dimension) for v in vectors]

        try:
            pending_backend = self._backend_factory.build(normalized)
        except IndexBuildError:
            raise
        except Exception as exc:
            LOG.warning("Vector backend construction failed (%s)", type(exc).__name__)
            raise IndexBuildError("Vector backend construction failed") from exc

        pending_index = VectorIndex(
            index_id=fingerprint,
            embedding_model=config.embedding_model,
            embedding_dimension=config.embedding_dimension,
            embedding_profile_id=config.embedding_profile_id,
            similarity_metric=config.similarity_metric,
            normalization_enabled=config.normalization_enabled,
            chunk_count=len(snapshot),
            document_count=len({c.document_id for c in snapshot}),
        )
        # Publish only after all steps have completed successfully.
        self._backend = pending_backend
        self._chunks = snapshot
        self._index = pending_index
        LOG.info("In-memory vector index ready: chunks=%d docs=%d", len(snapshot), pending_index.document_count)
        return pending_index

    def retrieve(self, index: VectorIndex, question: str, *, policy: RetrievalPolicy) -> RetrieverResult:
        if self._index is None or index is not self._index or self._backend is None:
            raise IndexNotFoundError("Vector index is unavailable or stale")
        if not question or not question.strip():
            raise RetrievalError("Question cannot be blank")
        try:
            raw_query = self._embedding_provider.embed_query(question)
        except EmbeddingError:
            raise
        except Exception as exc:
            LOG.warning("Query embedding provider failed (%s)", type(exc).__name__)
            raise EmbeddingError("Query embedding request failed") from exc
        normalized = _unit_vector(raw_query, self._index.embedding_dimension)
        try:
            # Get all positions for deterministic tie handling even at the k boundary.
            # V1 is a deliberately bounded single-user corpus; optimize later if needed.
            candidates = list(self._backend.search(normalized, top_k=len(self._chunks)))
        except Exception as exc:
            LOG.warning("Vector retrieval backend failed (%s)", type(exc).__name__)
            raise RetrievalError("Vector similarity search failed") from exc
        seen: set[int] = set()
        validated: list[tuple[float, DocumentChunk]] = []
        for match in candidates:
            position, score = match.position, match.similarity_score
            if not isinstance(position, int) or not (0 <= position < len(self._chunks)) or position in seen:
                raise RetrievalError("Invalid or duplicate vector position returned")
            seen.add(position)
            if not isinstance(score, (float, int)) or not math.isfinite(score):
                raise RetrievalError("Vector search produced an invalid score")
            if not -1.00001 <= score <= 1.00001:
                raise RetrievalError("Vector search score outside cosine range")
            cosine = max(-1.0, min(1.0, float(score)))
            if cosine >= policy.similarity_threshold:
                validated.append((cosine, self._chunks[position]))
        validated.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        hits = tuple(
            RetrievalHit(chunk=c, rank=i, similarity_score=s)
            for i, (s, c) in enumerate(validated[:policy.top_k], start=1)
        )
        return RetrieverResult(query_text=question, index_id=self._index.index_id, hits=hits,
                               candidate_k=policy.top_k, similarity_threshold=policy.similarity_threshold)

    def clear(self) -> None:
        self._index = None
        self._chunks = ()
        self._backend = None
