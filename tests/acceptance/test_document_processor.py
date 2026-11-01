"""RED: behavioral contract for DefaultDocumentProcessor (Step 5)."""
from pathlib import Path
import pytest
from pdf_rag.domain.models import PdfUpload
from pdf_rag.application.services import DefaultDocumentProcessor
from pdf_rag.application.config import IngestionPolicy, ChunkingPolicy
from pdf_rag.domain.errors import DocumentValidationError

pytestmark = pytest.mark.implementation
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def upload(file_name):
    return PdfUpload(file_name=file_name, content=(FIXTURES / file_name).read_bytes())


def process(*names, ingestion=None, chunking=None):
    return DefaultDocumentProcessor().process_uploads(
        [upload(name) for name in names],
        ingestion=ingestion or IngestionPolicy(),
        chunking=chunking or ChunkingPolicy(),
    )


def test_valid_pdf_is_accepted_with_original_page_numbers():
    report = process("two_pages.pdf")
    assert len(report.raw_documents) == 1
    doc = report.raw_documents[0]
    assert doc.page_count == 2
    assert "FIRST_PAGE_ANCHOR" in doc.pages[0].text
    assert "SECOND_PAGE_ANCHOR" in doc.pages[1].text
    assert report.rejected_files == ()
    assert all(c.page_number in {1, 2} for c in report.chunks)


@pytest.mark.parametrize("filename,expected_code", [
    ("encrypted.pdf", "encrypted_pdf"),
    ("image_only.pdf", "no_extractable_text"),
    ("blank.pdf", "no_extractable_text"),
    ("corrupted.pdf", "invalid_pdf"),
])
def test_invalid_and_unextractable_pdfs_are_rejected_without_chunks(filename, expected_code):
    report = process(filename)
    assert report.raw_documents == ()
    assert report.chunks == ()
    assert len(report.rejected_files) == 1
    assert report.rejected_files[0].error_code == expected_code
    assert report.rejected_files[0].file_name == filename


def test_mixed_batch_reports_failures_and_keeps_valid_document():
    report = process("one_page.pdf", "encrypted.pdf", "blank.pdf")
    assert [d.file_name for d in report.raw_documents] == ["one_page.pdf"]
    assert len(report.chunks) >= 1
    assert {r.file_name for r in report.rejected_files} == {"encrypted.pdf", "blank.pdf"}


def test_oversized_upload_rejected_without_crashing_other_uploads():
    report = process("one_page.pdf", "two_pages.pdf", ingestion=IngestionPolicy(max_file_size_bytes=2000))
    assert len(report.rejected_files) >= 1
    assert all(d.file_size_bytes <= 2000 for d in report.raw_documents)


def test_too_many_files_is_batch_level_validation_error():
    with pytest.raises(DocumentValidationError):
        process("one_page.pdf", "two_pages.pdf", ingestion=IngestionPolicy(max_files=1))


def test_total_upload_byte_limit_is_enforced():
    combined = len(upload("one_page.pdf").content) + len(upload("two_pages.pdf").content)
    with pytest.raises(DocumentValidationError):
        process("one_page.pdf", "two_pages.pdf", ingestion=IngestionPolicy(max_total_size_bytes=combined - 1))


def test_original_pdf_page_limit_is_enforced():
    with pytest.raises(DocumentValidationError):
        process("two_pages.pdf", ingestion=IngestionPolicy(max_total_pages=1))


def test_same_file_and_filename_deduplicated():
    report = process("one_page.pdf", "one_page.pdf")
    assert len(report.raw_documents) == 1
    assert len({c.chunk_id for c in report.chunks}) == len(report.chunks)


def test_cross_page_chunking_never_loses_original_page_provenance():
    report = process("two_pages.pdf", chunking=ChunkingPolicy(chunk_size=200, chunk_overlap=30))
    assert len(report.chunks) >= 4
    for c in report.chunks:
        source = report.raw_documents[0].pages[c.page_number - 1].text
        assert c.text in source
        assert not ("FIRST_PAGE_ANCHOR" in c.text and "SECOND_PAGE_ANCHOR" in c.text)
    assert set(c.page_number for c in report.chunks) == {1, 2}


def test_chunk_size_overlap_and_indexes_are_deterministic():
    policy = ChunkingPolicy(chunk_size=250, chunk_overlap=50)
    first = process("long_text.pdf", chunking=policy)
    second = process("long_text.pdf", chunking=policy)
    assert len(first.chunks) > 2
    assert first.chunks == second.chunks
    assert all(0 < len(c.text) <= 250 for c in first.chunks)
    assert [c.chunk_index for c in first.chunks] == list(range(len(first.chunks)))
    assert len({c.chunk_id for c in first.chunks}) == len(first.chunks)


def test_different_chunking_configuration_changes_chunk_identities():
    a = process("long_text.pdf", chunking=ChunkingPolicy(chunk_size=400, chunk_overlap=50))
    b = process("long_text.pdf", chunking=ChunkingPolicy(chunk_size=200, chunk_overlap=25))
    assert {c.chunk_id for c in a.chunks} != {c.chunk_id for c in b.chunks}
