from pdf_rag.domain import errors

def test_errors_are_application_specific_exceptions():
    names = [
        "ConfigurationError", "DocumentValidationError", "DocumentExtractionError",
        "ChunkingError", "EmbeddingError", "IndexBuildError", "IndexNotFoundError",
        "RetrievalError", "ContextAssemblyError", "AnswerGenerationError",
        "AnswerValidationError", "RateLimitError", "QuotaExhaustedError",
    ]
    for name in names:
        exc = getattr(errors, name)
        assert issubclass(exc, errors.RAGError)
        assert issubclass(exc, Exception)
