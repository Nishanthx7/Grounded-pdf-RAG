"""GREEN: independent PDF library verification, NOT processor implementation."""
from pathlib import Path
import pytest
pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_pypdf_extracts_original_pages_and_text():
    r = pypdf.PdfReader(FIXTURES / "two_pages.pdf")
    assert len(r.pages) == 2
    assert "FIRST_PAGE_ANCHOR" in r.pages[0].extract_text()
    assert "SECOND_PAGE_ANCHOR" in r.pages[1].extract_text()


def test_pypdf_cannot_extract_raster_only_pdf_without_ocr():
    r = pypdf.PdfReader(FIXTURES / "image_only.pdf")
    assert not (r.pages[0].extract_text() or "").strip()


def test_pypdf_detects_password_protected_document():
    r = pypdf.PdfReader(FIXTURES / "encrypted.pdf")
    assert r.is_encrypted


def test_pypdf_preserves_empty_page_without_text():
    r = pypdf.PdfReader(FIXTURES / "blank.pdf")
    assert len(r.pages) == 1
    assert not (r.pages[0].extract_text() or "").strip()
