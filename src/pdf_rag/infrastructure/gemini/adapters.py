"""Gemini adapters for text embeddings and structured answer generation.

Production factories lazily import clients; tests inject offline fakes.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
import logging
import time
import random
import re
from typing import Any, Sequence

from pdf_rag.application.config import GenerationPolicy
from pdf_rag.domain.errors import (
    ConfigurationError, EmbeddingError, RateLimitError, QuotaExhaustedError,
    AnswerGenerationError, AnswerValidationError, RAGError,
)
from pdf_rag.domain.models import AnswerDraft, AnswerStatus, GroundingContext, GenerationResult, TokenUsage

LOG = logging.getLogger(__name__)
DOCUMENT_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: question answering | query: "


# Diagnostic helpers report only error categories and numeric status codes.
# Never log exception messages: SDK errors can include request/response content.
_HTTP_STATUS_RE = re.compile(
    r"(?:\bHTTP\s*|\bstatus\s*[:=]?\s*|\bcode\s*[:=]?\s*|\bcontent:\s*|^\s*)"
    r"(400|401|403|404|408|409|413|422|429|500|502|503|504)\b",
    flags=re.IGNORECASE,
)

def _provider_diagnostic(exc: Exception) -> tuple[str, str]:
    """Return (numeric HTTP status or unknown, source exception type).

    Examine at most three chained exceptions. Exception text is inspected locally
    for status numbers but is never printed or written to logs.
    """
    current: BaseException | None = exc
    visited: set[int] = set()
    origin = type(exc).__name__
    for _ in range(3):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        origin = type(current).__name__
        for field in ("status_code", "code"):
            value = getattr(current, field, None)
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return str(value), origin
            if isinstance(value, str) and value.isdigit() and 100 <= int(value) <= 599:
                return value, origin
        # Extract *only* an HTTP status number, never log the message itself.
        found = _HTTP_STATUS_RE.search(str(current))
        if found:
            return found.group(1), origin
        current = current.__cause__ or current.__context__
    return "unknown", origin


def embedding_profile_id(*, model: str = "gemini-embedding-2", dimension: int = 768,
                         document_prefix: str = DOCUMENT_PREFIX, query_prefix: str = QUERY_PREFIX) -> str:
    """Stable fingerprint of the selected embedding representation and preprocessing."""
    document = json.dumps({"v": 1, "model": model, "dimension": dimension,
                           "document_prefix": document_prefix, "query_prefix": query_prefix},
                          sort_keys=True, separators=(",", ":"))
    return sha256(document.encode("utf-8")).hexdigest()


class GeminiEmbeddingProvider:
    """LangChain embedding-client adapter; no model client created for injected mocks.

    The default no-prefix mode is retained for existing Step 4 injected-client
    tests. Use `create_google_embedding_provider` for the recommended Gemini 2
    question-answering profile, which adds task instructions to the text.
    """

    def __init__(self, embedding_client: Any, *, embedding_model: str = "gemini-embedding-2",
                 dimension: int = 768, document_prefix: str = "", query_prefix: str = "",
                 batch_size: int = 32, max_attempts: int = 3, backoff_seconds: float = 0.25,
                 batch_interval_seconds: float = 0.0, rate_limit_backoff_seconds: float = 0.0,
                 sleep: Any = time.sleep) -> None:
        if (dimension < 1 or batch_size < 1 or batch_size > 100 or max_attempts < 1
                or backoff_seconds < 0 or batch_interval_seconds < 0
                or rate_limit_backoff_seconds < 0):
            raise ValueError("Invalid embedding adapter dimension or batch size")
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.batch_interval_seconds = batch_interval_seconds
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self._sleep = sleep
        self.embedding_profile_id = embedding_profile_id(
            model=embedding_model, dimension=dimension,
            document_prefix=document_prefix, query_prefix=query_prefix,
        )

    @staticmethod
    def _daily_quota_exceeded(exc: BaseException) -> bool:
        """Distinguish known non-transient daily-limit errors when indicated.

        Inspect messages locally, but never log them: SDK messages may contain
        request contents or credentials. A generic 429 is not assumed daily.
        """
        current: BaseException | None = exc
        visited: set[int] = set()
        for _ in range(5):
            if current is None or id(current) in visited:
                break
            visited.add(id(current))
            label = type(current).__name__.lower()
            message = str(current).lower()
            if ("quotaexhausted" in label or "daily quota" in message
                    or "requests per day" in message or "per_day" in message
                    or "limit: 0" in message):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _raise_safe(exc: Exception) -> None:
        """Classify nested provider failures without revealing raw responses."""
        status, _ = _provider_diagnostic(exc)
        if GeminiEmbeddingProvider._daily_quota_exceeded(exc):
            raise QuotaExhaustedError("Gemini embedding daily quota exhausted") from None
        if status == "429":
            raise RateLimitError("Gemini embedding rate or quota limit reached") from None
        raise EmbeddingError("Gemini embedding request failed") from None

    def _call(self, operation, *args, _request_label: str = "unknown", **kwargs):
        for attempt in range(self.max_attempts):
            try:
                return operation(*args, **kwargs)
            except Exception as exc:
                status, origin = _provider_diagnostic(exc)
                is_daily = self._daily_quota_exceeded(exc)
                LOG.warning(
                    "Gemini embedding diagnostic: request=%s attempt=%s/%s "
                    "wrapper=%s cause=%s http_status=%s daily_quota=%s",
                    _request_label, attempt + 1, self.max_attempts,
                    type(exc).__name__, origin, status, is_daily,
                )
                transient = (status in {"429", "408", "500", "502", "503", "504"}) and not is_daily
                if transient and attempt < self.max_attempts - 1:
                    delay = self.backoff_seconds * (2 ** attempt)
                    if status == "429":
                        delay = max(delay, self.rate_limit_backoff_seconds * (2 ** attempt))
                    LOG.warning(
                        "Gemini embedding retry: request=%s delay_seconds=%s",
                        _request_label, delay,
                    )
                    self._sleep(delay)
                    continue
                self._raise_safe(exc)
        raise EmbeddingError("Embedding retries exhausted")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(item, str) or not item.strip() for item in texts):
            raise EmbeddingError("Cannot embed an empty document chunk")
        output: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for offset in range(0, len(texts), self.batch_size):
            batch_no = offset // self.batch_size + 1
            if batch_no > 1 and self.batch_interval_seconds:
                LOG.info("Gemini embedding pacing: next_batch=%s/%s delay_seconds=%s",
                         batch_no, total_batches, self.batch_interval_seconds)
                self._sleep(self.batch_interval_seconds)
            batch = [self.document_prefix + s for s in texts[offset:offset+self.batch_size]]
            vectors = self._call(self.embedding_client.embed_documents, batch,
                                 _request_label=f"documents batch={batch_no}/{total_batches}",
                                 output_dimensionality=self.dimension)
            if not isinstance(vectors, (list, tuple)) or len(vectors) != len(batch):
                raise EmbeddingError("Embedding response count does not match request")
            output.extend(vectors)
            LOG.info("Gemini embedding completed: batch=%s/%s count=%s",
                     batch_no, total_batches, len(vectors))
        return output

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Cannot embed a blank query")
        return self._call(self.embedding_client.embed_query, self.query_prefix + text,
                          _request_label="query", output_dimensionality=self.dimension)


def create_google_embedding_provider(*, api_key: str | None = None, model: str = "gemini-embedding-2",
                                     dimension: int = 768, batch_size: int = 32,
                                     batch_interval_seconds: float = 25.0,
                                     rate_limit_backoff_seconds: float = 30.0) -> GeminiEmbeddingProvider:
    """Lazy production composition helper. Only called explicitly, never by tests.

    Caller should supply IndexBuildConfig embedding_profile_id from
    `provider.embedding_profile_id` and embedding_model from `provider.embedding_model`.
    """
    from dotenv import load_dotenv
    load_dotenv()
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ConfigurationError("Set GOOGLE_API_KEY or GEMINI_API_KEY before initializing Gemini embeddings")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    kwargs: dict[str, Any] = {"model": model, "output_dimensionality": dimension,
                              "google_api_key": key}
    client = GoogleGenerativeAIEmbeddings(**kwargs)
    return GeminiEmbeddingProvider(
        client, embedding_model=model, dimension=dimension,
        document_prefix=DOCUMENT_PREFIX, query_prefix=QUERY_PREFIX,
        batch_size=batch_size,
        batch_interval_seconds=batch_interval_seconds,
        rate_limit_backoff_seconds=rate_limit_backoff_seconds,
    )



def _extract_usage(raw: Any) -> TokenUsage:
    """Read provider-reported token metadata; never fabricate unavailable counts."""
    if raw is None:
        return TokenUsage()
    usage = getattr(raw, "usage_metadata", None)
    if usage is None and isinstance(raw, dict):
        usage = raw.get("usage_metadata")
    if not usage:
        response = getattr(raw, "response_metadata", None)
        if isinstance(response, dict):
            usage = response.get("token_usage") or response.get("usage_metadata")
    if not isinstance(usage, dict):
        return TokenUsage()

    def read(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    return TokenUsage(
        input_tokens=read("input_tokens", "prompt_token_count", "prompt_tokens"),
        output_tokens=read("output_tokens", "candidates_token_count", "completion_tokens"),
        total_tokens=read("total_tokens", "total_token_count"),
    )


_INLINE_CANONICAL_CITATION = re.compile(r"\[(S[1-9][0-9]*)\]")
_ANY_NUMERIC_CITATION = re.compile(r"\[S[0-9]+\]")


def _normalize_redundant_citations(
    result: GenerationResult, context: GroundingContext,
) -> GenerationResult:
    """Repair ONLY an omitted duplicate citation representation.

    Never overwrite conflicting nonempty lists, hallucinated IDs,
    duplicate structured IDs, malformed markers, or abstentions.
    The RAG pipeline independently enforces exact final consistency.
    """
    draft = result.draft
    if draft.status != AnswerStatus.ANSWERED:
        return result

    inline = tuple(dict.fromkeys(
        _INLINE_CANONICAL_CITATION.findall(draft.answer_text)
    ))
    structured = tuple(draft.citation_ids)
    allowed = set(context.allowed_citation_ids)

    # Refuse to "repair" malformed-looking references, e.g. [S01].
    raw_markers = _ANY_NUMERIC_CITATION.findall(draft.answer_text)
    if len(raw_markers) != len(_INLINE_CANONICAL_CITATION.findall(draft.answer_text)):
        return result

    if inline and not structured and set(inline).issubset(allowed):
        repaired = draft.model_copy(update={"citation_ids": inline})
        return result.model_copy(update={"draft": repaired})

    if not inline and structured and len(set(structured)) == len(structured):
        if set(structured).issubset(allowed):
            suffix = " ".join(f"[{cid}]" for cid in structured)
            repaired = draft.model_copy(update={
                "answer_text": draft.answer_text.rstrip() + " " + suffix,
            })
            return result.model_copy(update={"draft": repaired})

    return result


class GeminiStructuredAnswerGenerator:
    """Adapter for `build_qa_prompt() | chat.with_structured_output(...)`.

    An injected chain is supported for local deterministic acceptance tests.
    A production chain is model-bound at construction time; per-request policy
    changes to model/temperature/output limit require a new generator.
    """

    def __init__(
        self, chain: Any, *, bound_policy: GenerationPolicy | None = None,
        max_attempts: int = 3, backoff_seconds: float = 0.25,
        sleep: Any = time.sleep, jitter: Any = random.random,
    ) -> None:
        if max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("Invalid generation retry configuration")
        self.chain = chain
        self.bound_policy = bound_policy
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._jitter = jitter

    @staticmethod
    def _classify(exc: Exception) -> RAGError:
        if isinstance(exc, RAGError):
            return exc
        name = type(exc).__name__.lower()
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if str(code) in {"401", "403"} or "unauthenticated" in name or "permissiondenied" in name:
            return ConfigurationError("Gemini model credentials or access are invalid")
        text = str(exc).lower()
        if "quotaexhausted" in name or ("quota" in text and ("daily" in text or "limit: 0" in text)):
            return QuotaExhaustedError("Gemini generation quota exhausted")
        if str(code) == "429" or any(x in name for x in ("ratelimit", "toomanyrequests", "resourceexhausted")):
            return RateLimitError("Gemini answer generation rate limited")
        return AnswerGenerationError("Gemini answer generation request failed")

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        if isinstance(exc, QuotaExhaustedError):
            return False
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        if "quota" in text and ("daily" in text or "limit: 0" in text):
            return False
        return str(code) in {"429", "500", "502", "503", "504"} or any(
            term in name for term in ("ratelimit", "toomanyrequests", "resourceexhausted", "serviceunavailable")
        )

    @staticmethod
    def _parse_result(value: Any) -> GenerationResult:
        if isinstance(value, GenerationResult):
            return value
        raw = None
        parsed = value
        if isinstance(value, dict) and "parsed" in value:
            if value.get("parsing_error") is not None:
                raise AnswerValidationError("Gemini response failed structured output parsing")
            parsed = value["parsed"]
            raw = value.get("raw")
        try:
            draft = parsed if isinstance(parsed, AnswerDraft) else AnswerDraft.model_validate(parsed)
        except Exception:
            raise AnswerValidationError("Gemini returned an invalid answer schema") from None
        return GenerationResult(draft=draft, token_usage=_extract_usage(raw))

    def generate(
        self, question: str, context: GroundingContext, *, policy: GenerationPolicy,
    ) -> GenerationResult:
        if not question or not question.strip():
            raise AnswerValidationError("Question must not be blank")
        if self.bound_policy is not None:
            # Context budget is validated by the assembler; changing it does
            # not change the already-constructed LLM chain.
            for field in ("llm_model", "temperature", "max_output_tokens", "prompt_version"):
                if getattr(policy, field) != getattr(self.bound_policy, field):
                    raise ConfigurationError("Generation settings require a new Gemini generator")
        if policy.prompt_version != "v1":
            raise ConfigurationError("Unsupported prompt version")
        payload = {"question": question, "context": context.context_text}
        for attempt in range(self.max_attempts):
            try:
                return _normalize_redundant_citations(self._parse_result(self.chain.invoke(payload)), context)
            except (AnswerValidationError, ConfigurationError):
                raise
            except Exception as exc:
                typed = self._classify(exc)
                if self._is_transient(exc) and attempt + 1 < self.max_attempts:
                    self._sleep(self.backoff_seconds * (2 ** attempt) * (1 + 0.10 * self._jitter()))
                    continue
                LOG.warning("Gemini answer call failed (%s)", type(exc).__name__)
                raise typed from None
        raise AnswerGenerationError("Gemini response attempts exhausted")


def create_google_answer_generator(
    *, policy: GenerationPolicy | None = None, api_key: str | None = None,
    max_attempts: int = 3,
) -> GeminiStructuredAnswerGenerator:
    """Opt-in production factory for Gemini + LCEL structured Pydantic output.

    Never invoked during import or offline tests. No question or PDF content is
    logged. The API key stays out of all domain models and prompts.
    """
    from dotenv import load_dotenv
    load_dotenv()
    policy = policy or GenerationPolicy()
    if policy.prompt_version != "v1":
        raise ConfigurationError("Unsupported prompt version")
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ConfigurationError("Set GOOGLE_API_KEY or GEMINI_API_KEY to use Gemini generation")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pdf_rag.infrastructure.gemini.prompt import build_qa_prompt
    chat = ChatGoogleGenerativeAI(
        model=policy.llm_model,
        temperature=policy.temperature,
        max_output_tokens=policy.max_output_tokens,
        google_api_key=key,
    )
    structured = chat.with_structured_output(
        AnswerDraft, method="json_schema", include_raw=True,
    )
    return GeminiStructuredAnswerGenerator(
        build_qa_prompt() | structured,
        bound_policy=policy,
        max_attempts=max_attempts,
    )
