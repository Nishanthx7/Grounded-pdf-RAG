"""Step 3 exception hierarchy, unchanged."""
class RAGError(Exception):
    """Base exception for application failures."""

class ConfigurationError(RAGError): pass
class DocumentValidationError(RAGError): pass
class DocumentExtractionError(RAGError): pass
class ChunkingError(RAGError): pass
class EmbeddingError(RAGError): pass
class IndexBuildError(RAGError): pass
class IndexNotFoundError(RAGError): pass
class RetrievalError(RAGError): pass
class ContextAssemblyError(RAGError): pass
class AnswerGenerationError(RAGError): pass
class AnswerValidationError(RAGError): pass
class RateLimitError(RAGError): pass
class QuotaExhaustedError(RAGError): pass
