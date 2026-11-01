"""FAISS flat inner-product adapter; inputs are unit normalized for cosine search."""
from __future__ import annotations
import math
from typing import Sequence
from pdf_rag.application.ports import VectorMatch, VectorSearchBackend
from pdf_rag.domain.errors import IndexBuildError, RetrievalError


class FaissVectorSearchBackend:
    def __init__(self, index, dimension: int, count: int) -> None:
        self._index = index
        self._dimension = dimension
        self._count = count

    def search(self, query_vector: Sequence[float], *, top_k: int) -> list[VectorMatch]:
        import numpy as np
        if not isinstance(top_k, int) or top_k < 1:
            raise RetrievalError("top_k must be positive")
        try:
            query = np.asarray(query_vector, dtype=np.float64)
            if query.ndim != 1 or query.shape[0] != self._dimension or not np.isfinite(query).all():
                raise ValueError("Invalid query vector")
            norm = float(np.linalg.norm(query))
            if norm == 0 or not math.isfinite(norm):
                raise ValueError("Invalid query vector norm")
            query = (query / norm).astype("float32").reshape(1, -1)
            scores, positions = self._index.search(query, min(top_k, self._count))
        except (ValueError, TypeError, OverflowError) as exc:
            raise RetrievalError("Invalid query vector") from exc
        return [VectorMatch(position=int(p), similarity_score=float(s))
                for p, s in zip(positions[0], scores[0]) if p >= 0]


class FaissVectorSearchBackendFactory:
    def build(self, vectors: Sequence[Sequence[float]]) -> VectorSearchBackend:
        import faiss
        import numpy as np
        try:
            arr = np.asarray(vectors, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
                raise ValueError("No valid vectors")
            if not np.isfinite(arr).all():
                raise ValueError("Nonfinite vectors")
            norms = np.linalg.norm(arr, axis=1)
            if not np.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("Zero or invalid vector norms")
            normalized = (arr / norms[:, None]).astype("float32")
            db = faiss.IndexFlatIP(int(arr.shape[1]))
            db.add(np.ascontiguousarray(normalized))
        except (ValueError, TypeError, OverflowError) as exc:
            raise IndexBuildError("Invalid vectors supplied to FAISS") from exc
        return FaissVectorSearchBackend(db, int(arr.shape[1]), int(arr.shape[0]))
