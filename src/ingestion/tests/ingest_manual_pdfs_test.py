import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.ingestion.ingest_manual_pdfs"
# ==============================================================================
MODULE_PATH = "ingestion.ingest_manual_pdfs"


@pytest.fixture
def target_module():
    """Dynamically imports the target module."""
    import importlib

    return importlib.import_module(MODULE_PATH)


# --- Tests for PDF Extraction Logic ---


@patch("pypdf.PdfReader")
def test_extract_text_pypdf_success(mock_pdf_reader, target_module):
    """Test successful text extraction using pypdf."""
    # Setup mock pages
    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = "Page 1 text."
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = "   Page 2 text.   "
    mock_page_3 = MagicMock()
    mock_page_3.extract_text.return_value = None  # Should be handled safely

    mock_instance = MagicMock()
    mock_instance.pages = [mock_page_1, mock_page_2, mock_page_3]
    mock_pdf_reader.return_value = mock_instance

    result = target_module.extract_text_pypdf("dummy.pdf")

    assert result == "Page 1 text.\n\nPage 2 text."
    mock_pdf_reader.assert_called_once_with("dummy.pdf")


@patch("pdfminer.high_level.extract_text")
def test_extract_text_pdfminer_success(mock_extract, target_module):
    """Test successful text extraction using pdfminer."""
    mock_extract.return_value = "Miner text extracted."

    result = target_module.extract_text_pdfminer("dummy.pdf")

    assert result == "Miner text extracted."
    mock_extract.assert_called_once_with("dummy.pdf")


@patch(f"{MODULE_PATH}.extract_text_pypdf")
@patch(f"{MODULE_PATH}.extract_text_pdfminer")
def test_extract_text_routing_success(mock_miner, mock_pypdf, target_module):
    """Test that extract_text prefers pypdf when it works."""
    mock_pypdf.return_value = "Valid text"

    result = target_module.extract_text("dummy.pdf")

    assert result == "Valid text"
    mock_pypdf.assert_called_once()
    mock_miner.assert_not_called()


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.extract_text_pypdf")
@patch(f"{MODULE_PATH}.extract_text_pdfminer")
def test_extract_text_fallback_on_empty(
    mock_miner, mock_pypdf, mock_logger, target_module
):
    """Test fallback to pdfminer if pypdf returns empty text."""
    mock_pypdf.return_value = "   \n  "  # Only whitespace
    mock_miner.return_value = "Fallback text"

    result = target_module.extract_text("dummy.pdf")

    assert result == "Fallback text"
    mock_pypdf.assert_called_once()
    mock_miner.assert_called_once()


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.extract_text_pypdf")
@patch(f"{MODULE_PATH}.extract_text_pdfminer")
def test_extract_text_fallback_on_exception(
    mock_miner, mock_pypdf, mock_logger, target_module
):
    """Test fallback to pdfminer if pypdf raises an exception."""
    mock_pypdf.side_effect = Exception("Corrupt PDF")
    mock_miner.return_value = "Fallback text"

    result = target_module.extract_text("dummy.pdf")

    assert result == "Fallback text"
    mock_logger.warning.assert_called_once()
    assert "pypdf failed" in mock_logger.warning.call_args[0][0]


# --- Tests for main() ---


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.makedirs")
def test_main_no_pdfs_found(mock_makedirs, mock_listdir, mock_logger, target_module):
    """Test that main stops early if no PDFs are found."""
    mock_listdir.return_value = ["not_a_pdf.txt", "image.png"]

    target_module.main()

    mock_makedirs.assert_called_once_with(target_module.OUTPUT_DIR, exist_ok=True)
    mock_logger.warning.assert_called_once()
    assert "No PDFs found" in mock_logger.warning.call_args[0][0]


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.extract_text")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.makedirs")
def test_main_skips_empty_extraction(
    mock_makedirs, mock_listdir, mock_extract, mock_logger, target_module
):
    """Test that files yielding no text are skipped and not saved."""
    mock_listdir.return_value = ["empty.pdf"]
    mock_extract.return_value = "   "  # Whitespace only

    m_open = mock_open()
    with patch("builtins.open", m_open):
        target_module.main()

    m_open.assert_not_called()
    mock_logger.warning.assert_called_with(
        "No text extracted from empty.pdf — skipping"
    )


@patch(f"{MODULE_PATH}.extract_text")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.makedirs")
def test_main_happy_path_with_mapping(
    mock_makedirs, mock_listdir, mock_extract, target_module
):
    """Test successful processing of a mapped PDF and an unmapped PDF."""

    # regulamin_studiow.pdf is in the URL_MAP, unknown_file.pdf is not
    mock_listdir.return_value = ["regulamin_studiow.pdf", "unknown_file.pdf"]

    # Mock text extraction to return something valid
    mock_extract.return_value = "Extracted PDF content"

    m_open = mock_open()
    with patch("builtins.open", m_open):
        target_module.main()

    # Assertions for the mapped file
    mapped_url = target_module.URL_MAP["regulamin_studiow.pdf"]
    expected_mapped_filename = (
        mapped_url.replace("https://", "").replace("/", "_").strip("_")[:200]
    )
    expected_mapped_path = os.path.join(
        target_module.OUTPUT_DIR, f"{expected_mapped_filename}.txt"
    )

    # Assertions for the unmapped file (fallback to file://)
    expected_unmapped_url = "file://unknown_file.pdf"
    expected_unmapped_filename = (
        expected_unmapped_url.replace("https://", "").replace("/", "_").strip("_")[:200]
    )
    expected_unmapped_path = os.path.join(
        target_module.OUTPUT_DIR, f"{expected_unmapped_filename}.txt"
    )

    # Verify files were opened
    assert m_open.call_count == 2
    m_open.assert_any_call(expected_mapped_path, "w", encoding="utf-8")
    m_open.assert_any_call(expected_unmapped_path, "w", encoding="utf-8")

    # Verify the correct content format was written
    expected_mapped_content = f"URL: {mapped_url}\n\nExtracted PDF content"
    expected_unmapped_content = f"URL: {expected_unmapped_url}\n\nExtracted PDF content"

    # Check that both contents were written
    handle = m_open()
    write_calls = [call[0][0] for call in handle.write.call_args_list]
    assert expected_mapped_content in write_calls
    assert expected_unmapped_content in write_calls
