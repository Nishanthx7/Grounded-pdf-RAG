"""Bounded, deterministic assembly of untrusted retrieval evidence.

The input is validated provenance from the retrieval contract. This module does
not know about Gemini, LangChain, FAISS, or Streamlit.
"""
from __future__ import annotations

import json
from typing import Callable

from pdf_rag.domain.errors import ContextAssemblyError
from pdf_rag.domain.models import GroundingContext, GroundingEvidence, RetrieverResult


def _utf8_upper_bound(text: str) -> int:
    """Conservative budget proxy; *not* a measured provider token count.

    Using UTF-8 bytes as a deliberately conservative limit is a V1 offline
    fallback when no provider tokenizer is available. context_token_count
    remains None. This budget covers retrieved context, not the system prompt.
    """
    return len(text.encode("utf-8"))


class GroundingContextAssembler:
    """Produce only citation-bearing evidence whose formatted text fits a budget."""

    def __init__(self, size_estimator: Callable[[str], int] = _utf8_upper_bound) -> None:
        self._size_estimator = size_estimator

    def assemble(self, retrieval: RetrieverResult, *, context_token_budget: int) -> GroundingContext:
        if context_token_budget < 1:
            raise ContextAssemblyError("Context budget must be positive")
        if not retrieval.hits:
            raise ContextAssemblyError("Cannot assemble an empty retrieval result")

        entries: list[GroundingEvidence] = []
        sections: list[str] = []
        accepted_ids: set[str] = set()
        for hit in retrieval.hits:
            source = hit.chunk
            if source.chunk_id in accepted_ids:
                raise ContextAssemblyError("Duplicate chunk in grounding result")
            citation_id = f"S{len(entries) + 1}"
            # JSON-quoted filename prevents line breaks and delimiter-like
            # characters in user-controlled filenames from altering the header.
            header = f"[{citation_id}] file={json.dumps(source.file_name, ensure_ascii=True)} page={source.page_number}\n"
            separator = "\n\n" if sections else ""
            prefix = separator + header
            used = self._size_estimator("".join(sections))
            available = context_token_budget - used
            if self._size_estimator(prefix + source.text) <= available:
                excerpt = source.text
            else:
                # Reserve a visible truncation marker; never silently pretend
                # the full chunk was included. Do not include tiny fragments.
                suffix = " … [truncated]"
                if self._size_estimator(prefix + suffix) >= available:
                    continue
                lo, hi = 0, len(source.text)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._size_estimator(prefix + source.text[:mid] + suffix) <= available:
                        lo = mid
                    else:
                        hi = mid - 1
                excerpt = source.text[:lo].rstrip()
                if len(excerpt.strip()) < 8:
                    continue
                # rstrip might reduce size; safe to keep the selected prefix.
                excerpt += suffix
            section = prefix + excerpt
            if self._size_estimator("".join(sections) + section) > context_token_budget:
                raise ContextAssemblyError("Context budget calculation failed")
            sections.append(section)
            entries.append(GroundingEvidence(
                citation_id=citation_id, chunk_id=source.chunk_id,
                document_id=source.document_id, file_name=source.file_name,
                page_number=source.page_number, text=excerpt,
            ))
            accepted_ids.add(source.chunk_id)
            # If truncated, remaining space should be too small for further
            # citations; still try following chunks only if they fit.
        if not entries:
            raise ContextAssemblyError("Context budget cannot hold usable evidence")
        return GroundingContext(
            context_text="".join(sections),
            evidence=tuple(entries),
            allowed_citation_ids=tuple(e.citation_id for e in entries),
            # The byte approximation must not be reported as provider tokens.
            context_token_count=None,
        )
