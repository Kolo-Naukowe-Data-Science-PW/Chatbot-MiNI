import os
from unittest.mock import MagicMock, mock_open, patch

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.ingestion.complex_files"
# ==============================================================================
MODULE_PATH = "ingestion.describe_files"


@patch(f"{MODULE_PATH}.pd.read_excel")
@patch(f"{MODULE_PATH}.pd.ExcelFile")
def test_process_xlsx_success(mock_excel_file, mock_read_excel):
    """Test successful extraction of text from an Excel file."""
    # 1. Setup the mocks
    mock_xls_instance = MagicMock()
    mock_xls_instance.sheet_names = ["Sheet1", "Sheet2"]
    mock_excel_file.return_value = mock_xls_instance

    mock_df = MagicMock()
    mock_df.to_markdown.return_value = "| mocked | data |"
    mock_read_excel.return_value = mock_df

    # 2. Call the function
    # We need to import the module here dynamically to avoid import errors
    # if the module path isn't set up right in the environment yet
    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    result = target_module.process_xlsx("dummy/path/data.xlsx")

    # 3. Assertions
    assert "Source: Excel file data.xlsx" in result
    assert "Sheet: Sheet1" in result
    assert "Sheet: Sheet2" in result
    assert "| mocked | data |" in result
    assert mock_read_excel.call_count == 2


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.pd.ExcelFile")
def test_process_xlsx_exception(mock_excel_file, mock_logger):
    """Test exception handling when Excel processing fails."""
    mock_excel_file.side_effect = Exception("Corrupt file")

    import importlib

    target_module = importlib.import_module(MODULE_PATH)
    result = target_module.process_xlsx("bad.xlsx")

    assert result is None
    mock_logger.error.assert_called_once()
    assert (
        "Error processing bad.xlsx: Corrupt file" in mock_logger.error.call_args[0][0]
    )


@patch(f"{MODULE_PATH}.Document")
def test_process_docx_success(mock_document_class):
    """Test successful extraction of text from a Word document."""
    mock_doc_instance = MagicMock()
    # Create mock paragraphs. The second one is whitespace and should be skipped.
    mock_doc_instance.paragraphs = [
        MagicMock(text="First paragraph."),
        MagicMock(text="   "),
        MagicMock(text="Second paragraph."),
    ]
    mock_document_class.return_value = mock_doc_instance

    import importlib

    target_module = importlib.import_module(MODULE_PATH)
    result = target_module.process_docx("dummy/path/doc.docx")

    assert "Source: Word document doc.docx" in result
    assert "First paragraph." in result
    assert "Second paragraph." in result
    assert "   " not in result.replace("\n", "")  # Empty paragraph shouldn't be joined


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.Document")
def test_process_docx_exception(mock_document_class, mock_logger):
    """Test exception handling when Word processing fails."""
    mock_document_class.side_effect = Exception("File locked")

    import importlib

    target_module = importlib.import_module(MODULE_PATH)
    result = target_module.process_docx("locked.docx")

    assert result is None
    mock_logger.error.assert_called_once()
    assert (
        "Error processing locked.docx: File locked" in mock_logger.error.call_args[0][0]
    )


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.json.dump")
def test_save_text_and_meta(mock_json_dump, mock_logger):
    """Test saving text and JSON metadata correctly."""
    m_open = mock_open()

    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    with patch("builtins.open", m_open):
        target_module.save_text_and_meta("test_file.xlsx", "extracted content")

    # Should open twice: once for .txt, once for .json
    assert m_open.call_count == 2

    txt_path = os.path.join(target_module.OUTPUT_DIR, "test_file.xlsx.txt")
    meta_path = os.path.join(target_module.OUTPUT_DIR, "test_file.xlsx.json")

    # Check calls to open()
    m_open.assert_any_call(txt_path, "w", encoding="utf-8")
    m_open.assert_any_call(meta_path, "w", encoding="utf-8")

    # Check what was written to the text file
    m_open().write.assert_called_once_with("extracted content")

    # Check JSON dump
    mock_json_dump.assert_called_once()
    assert mock_json_dump.call_args[0][0] == {"source_url": "test_file.xlsx"}

    mock_logger.info.assert_called_once_with("Processed: test_file.xlsx")


def test_main_config_disabled():
    """Test that main() skips processing if config is disabled."""
    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    # Temporarily modify the module's loaded config dict
    with patch.dict(target_module.config, {"process_complex_files": False}):
        with patch(f"{MODULE_PATH}.logger") as mock_logger:
            target_module.main()

            mock_logger.info.assert_called_once()
            assert "SKIP:" in mock_logger.info.call_args[0][0]


def test_main_input_dir_missing():
    """Test that main() stops gracefully if the input directory is missing."""
    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    with patch.dict(target_module.config, {"process_complex_files": True}):
        with patch(f"{MODULE_PATH}.os.makedirs"):
            with patch(f"{MODULE_PATH}.os.path.exists", return_value=False):
                with patch(f"{MODULE_PATH}.logger") as mock_logger:
                    target_module.main()

                    mock_logger.warning.assert_called_once()
                    assert (
                        "Input directory does not exist"
                        in mock_logger.warning.call_args[0][0]
                    )


@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}.os.makedirs")
@patch(f"{MODULE_PATH}.process_xlsx")
@patch(f"{MODULE_PATH}.process_docx")
@patch(f"{MODULE_PATH}.save_text_and_meta")
def test_main_happy_path(
    mock_save,
    mock_process_docx,
    mock_process_xlsx,
    mock_makedirs,
    mock_exists,
    mock_listdir,
):
    """Test that main() processes supported files and ignores others."""
    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    # Setup mocks
    mock_exists.return_value = True
    # Provide one Excel, one Word, and one unsupported file
    mock_listdir.return_value = ["data.xlsx", "doc.docx", "image.png"]

    mock_process_xlsx.return_value = "excel text"
    mock_process_docx.return_value = "word text"

    with patch.dict(target_module.config, {"process_complex_files": True}):
        target_module.main()

    mock_makedirs.assert_called_once_with(target_module.OUTPUT_DIR, exist_ok=True)

    # Processors should only be called for matching extensions
    mock_process_xlsx.assert_called_once_with(
        os.path.join(target_module.INPUT_DIR, "data.xlsx")
    )
    mock_process_docx.assert_called_once_with(
        os.path.join(target_module.INPUT_DIR, "doc.docx")
    )

    # Save should be called twice (ignoring image.png)
    assert mock_save.call_count == 2
    mock_save.assert_any_call("data.xlsx", "excel text")
    mock_save.assert_any_call("doc.docx", "word text")
