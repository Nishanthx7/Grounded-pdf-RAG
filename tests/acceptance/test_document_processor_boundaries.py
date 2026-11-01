"""Additional RED-first ingestion acceptance cases for Step 5 phase 1.

The module monkeypatches the splitter only where necessary to test the PDF
processor boundary without the external LangChain installation.
"""

from pathlib import Path

import pytest

from pdf_rag.application.config import ChunkingPolicy, IngestionPolicy
from pdf_rag.application.services import DefaultDocumentProcessor
from pdf_rag.domain.errors import ChunkingError, DocumentValidationError
from pdf_rag.domain.models import PdfUpload
import pdf_rag.infrastructure.pdf.processor as implementation


pytestmark = pytest.mark.implementation
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def valid_upload(name="one_page.pdf"):
    return PdfUpload(file_name=name, content=(FIXTURES / "one_page.pdf").read_bytes())


class IdentitySplitter:
    """Controlled test double that does not change its input text."""

    def split_text(self, text):
        return [text]


@pytest.fixture
def processor_without_external_splitter(monkeypatch):
    monkeypatch.setattr(implementation, "_new_splitter", lambda policy: IdentitySplitter())
    return DefaultDocumentProcessor()


def test_exact_duplicate_document_ids_are_stable(processor_without_external_splitter):
    first = processor_without_external_splitter.process_uploads(
        [valid_upload()], ingestion=IngestionPolicy(), chunking=ChunkingPolicy()
    )
    second = processor_without_external_splitter.process_uploads(
        [valid_upload()], ingestion=IngestionPolicy(), chunking=ChunkingPolicy()
    )
    assert first == second
    assert first.raw_documents[0].document_id == second.raw_documents[0].document_id
    assert len(first.raw_documents) == 1


def test_identical_bytes_with_different_filenames_remain_distinct(processor_without_external_splitter):
    report = processor_without_external_splitter.process_uploads(
        [valid_upload("a.pdf"), valid_upload("b.pdf")],
        ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
    )
    assert len(report.raw_documents) == 2
    assert report.raw_documents[0].content_sha256 == report.raw_documents[1].content_sha256
    assert report.raw_documents[0].document_id != report.raw_documents[1].document_id
    assert len({chunk.chunk_id for chunk in report.chunks}) == 2


def test_path_components_are_stripped_from_display_name(processor_without_external_splitter):
    report = processor_without_external_splitter.process_uploads(
        [valid_upload("../private\\report.PDF")],
        ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
    )
    assert report.raw_documents[0].file_name == "report.PDF"
    assert report.chunks[0].file_name == "report.PDF"


def test_unsupported_extension_is_rejected_before_pdf_parsing(processor_without_external_splitter, monkeypatch):
    def no_parse(*args, **kwargs):
        raise AssertionError("invalid extension should not reach pypdf")
    monkeypatch.setattr(implementation, "PdfReader", no_parse)
    report = processor_without_external_splitter.process_uploads(
        [valid_upload("report.txt")], ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
    )
    assert report.raw_documents == ()
    assert report.rejected_files[0].error_code == "unsupported_file_type"


def test_invalid_file_name_is_rejected(processor_without_external_splitter):
    report = processor_without_external_splitter.process_uploads(
        [valid_upload("evil\nname.pdf")],
        ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
    )
    assert report.rejected_files[0].error_code == "invalid_filename"
    assert report.chunks == ()


def test_empty_upload_collection_is_batch_error(processor_without_external_splitter):
    with pytest.raises(DocumentValidationError, match="At least one"):
        processor_without_external_splitter.process_uploads(
            [], ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
        )


def test_batch_limits_checked_before_pdf_parser(processor_without_external_splitter, monkeypatch):
    def no_parse(*args, **kwargs):
        raise AssertionError("batch violation must not parse PDFs")
    monkeypatch.setattr(implementation, "PdfReader", no_parse)
    with pytest.raises(DocumentValidationError):
        processor_without_external_splitter.process_uploads(
            [valid_upload(), valid_upload()],
            ingestion=IngestionPolicy(max_files=1), chunking=ChunkingPolicy(),
        )


def test_page_level_extraction_error_is_reported_and_good_page_remains(processor_without_external_splitter, monkeypatch):
    class GoodPage:
        def extract_text(self):
            return "Supported evidence on this page."

    class BadPage:
        def extract_text(self):
            raise RuntimeError("do not expose this sensitive parser exception")

    class FakeReader:
        is_encrypted = False
        pages = [GoodPage(), BadPage()]

    monkeypatch.setattr(implementation, "PdfReader", lambda *args, **kwargs: FakeReader())
    report = processor_without_external_splitter.process_uploads(
        [valid_upload()], ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
    )
    assert len(report.raw_documents) == 1
    doc = report.raw_documents[0]
    assert doc.page_count == 2
    assert doc.pages[1].text == ""
    assert len(doc.extraction_issues) == 1
    assert doc.extraction_issues[0].page_number == 2
    assert "sensitive parser" not in doc.extraction_issues[0].message
    assert len(report.chunks) == 1
    assert report.chunks[0].page_number == 1


def test_splitter_output_not_from_source_is_rejected(monkeypatch):
    class ForgingSplitter:
        def split_text(self, text):
            return ["INVENTED BY SPLITTER"]

    monkeypatch.setattr(implementation, "_new_splitter", lambda policy: ForgingSplitter())
    with pytest.raises(ChunkingError, match="provenance"):
        DefaultDocumentProcessor().process_uploads(
            [valid_upload()], ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
        )


def test_splitter_exception_fails_explicitly(monkeypatch):
    class FailingSplitter:
        def split_text(self, text):
            raise RuntimeError("splitter exploded")

    monkeypatch.setattr(implementation, "_new_splitter", lambda policy: FailingSplitter())
    with pytest.raises(ChunkingError, match="Text splitting failed"):
        DefaultDocumentProcessor().process_uploads(
            [valid_upload()], ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
        )


def test_none_text_is_an_empty_page_not_extraction_error(processor_without_external_splitter, monkeypatch):
    class FakePage:
        def __init__(self, value):
            self.value = value
        def extract_text(self):
            return self.value
    class FakeReader:
        is_encrypted = False
        pages = [FakePage(None), FakePage("Evidence from page two.")]
    monkeypatch.setattr(implementation, "PdfReader", lambda *args, **kwargs: FakeReader())
    report = processor_without_external_splitter.process_uploads(
        [valid_upload()], ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
    )
    assert report.raw_documents[0].extraction_issues == ()
    assert report.raw_documents[0].pages[0].text == ""
    assert report.chunks[0].page_number == 2


def test_every_page_extraction_exception_rejects_document_explicitly(processor_without_external_splitter, monkeypatch):
    class FailingPage:
        def extract_text(self):
            raise ValueError("boom")
    class FakeReader:
        is_encrypted = False
        pages = [FailingPage()]
    monkeypatch.setattr(implementation, "PdfReader", lambda *args, **kwargs: FakeReader())
    report = processor_without_external_splitter.process_uploads(
        [valid_upload()], ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
    )
    assert report.raw_documents == ()
    assert report.chunks == ()
    assert report.rejected_files[0].error_code == "extraction_failed"


def test_non_string_splitter_output_raises_chunking_error(monkeypatch):
    class InvalidSplitter:
        def split_text(self, text):
            return [None]
    monkeypatch.setattr(implementation, "_new_splitter", lambda policy: InvalidSplitter())
    with pytest.raises(ChunkingError, match="invalid output"):
        DefaultDocumentProcessor().process_uploads(
            [valid_upload()], ingestion=IngestionPolicy(), chunking=ChunkingPolicy(),
        )
