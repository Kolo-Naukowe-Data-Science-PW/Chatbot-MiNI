import os
from unittest.mock import mock_open, patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.ingestion.ingest_manual_xlsx"
# ==============================================================================
MODULE_PATH = "ingestion.ingest_manual_xlsx"


@pytest.fixture
def target_module():
    """Dynamically imports the target module."""
    import importlib

    return importlib.import_module(MODULE_PATH)


# --- Tests for extract_text() ---
@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.pd.ExcelFile")
def test_extract_text_exception(mock_excel_file, mock_logger, target_module):
    """Test that pandas exceptions are caught and logged."""
    mock_excel_file.side_effect = Exception("Corrupt XLSX file")

    result = target_module.extract_text("broken.xlsx")

    assert result == ""
    mock_logger.error.assert_called_once()
    assert (
        "Failed to read broken.xlsx: Corrupt XLSX file"
        in mock_logger.error.call_args[0][0]
    )


# --- Tests for main() ---


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.makedirs")
def test_main_no_xlsx_found(mock_makedirs, mock_listdir, mock_logger, target_module):
    """Test that main stops early if no XLSX files are found."""
    mock_listdir.return_value = ["document.pdf", "image.png"]

    target_module.main()

    mock_makedirs.assert_called_once_with(target_module.OUTPUT_DIR, exist_ok=True)
    mock_logger.warning.assert_called_once()
    assert "No XLSX files found" in mock_logger.warning.call_args[0][0]


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.extract_text")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.makedirs")
def test_main_skips_empty_extraction(
    mock_makedirs, mock_listdir, mock_extract, mock_logger, target_module
):
    """Test that files yielding no text are skipped and not saved."""
    mock_listdir.return_value = ["empty.xlsx"]
    mock_extract.return_value = "   "  # Whitespace only

    m_open = mock_open()
    with patch("builtins.open", m_open):
        target_module.main()

    m_open.assert_not_called()
    mock_logger.warning.assert_called_with(
        "No text extracted from empty.xlsx — skipping"
    )


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.extract_text")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.makedirs")
def test_main_happy_path_with_mapping(
    mock_makedirs, mock_listdir, mock_extract, mock_logger, target_module
):
    """Test processing of a mapped XLSX and an unmapped fallback XLSX."""

    # "CSIS_WINTER_25Z (1).xlsx" is in the URL_MAP, "unmapped_schedule.xlsx" is not
    mock_listdir.return_value = ["CSIS_WINTER_25Z (1).xlsx", "unmapped_schedule.xlsx"]
    mock_extract.return_value = "Markdown Table Content"

    m_open = mock_open()
    with patch("builtins.open", m_open):
        target_module.main()

    # Assertions for the mapped file
    mapped_url = target_module.URL_MAP["CSIS_WINTER_25Z (1).xlsx"]
    expected_mapped_filename = (
        mapped_url.replace("https://", "").replace("/", "_").strip("_")[:200]
    )
    expected_mapped_path = os.path.join(
        target_module.OUTPUT_DIR, f"{expected_mapped_filename}.txt"
    )

    # Assertions for the unmapped file (fallback to file://)
    expected_unmapped_url = "file://unmapped_schedule.xlsx"
    expected_unmapped_filename = (
        expected_unmapped_url.replace("https://", "").replace("/", "_").strip("_")[:200]
    )
    expected_unmapped_path = os.path.join(
        target_module.OUTPUT_DIR, f"{expected_unmapped_filename}.txt"
    )

    # Verify unmapped file triggered a warning
    mock_logger.warning.assert_any_call(
        "unmapped_schedule.xlsx has no URL mapping — using filename as fallback URL"
    )

    # Verify files were opened
    assert m_open.call_count == 2
    m_open.assert_any_call(expected_mapped_path, "w", encoding="utf-8")
    m_open.assert_any_call(expected_unmapped_path, "w", encoding="utf-8")

    # Check what was actually written
    handle = m_open()
    write_calls = [call.args[0] for call in handle.write.call_args_list]

    assert f"URL: {mapped_url}\n\nMarkdown Table Content" in write_calls
    assert f"URL: {expected_unmapped_url}\n\nMarkdown Table Content" in write_calls
