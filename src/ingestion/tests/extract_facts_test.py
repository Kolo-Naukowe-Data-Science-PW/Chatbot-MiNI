from unittest.mock import MagicMock, mock_open, patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.ingestion.extract_facts"
# ==============================================================================
MODULE_PATH = "ingestion.extract_facts"


@pytest.fixture
def target_module():
    """Dynamically imports the target module to avoid early-load environment errors."""
    import importlib

    return importlib.import_module(MODULE_PATH)


## Tests for extract_facts_list


def test_extract_facts_list_llm_disabled(target_module):
    """Test that when use_llm_for_facts is False, it returns the raw text."""
    with patch.dict(target_module.config, {"use_llm_for_facts": False}):
        result = target_module.extract_facts_list("  Raw text content  ", "test.txt")
        assert result == ["Raw text content"]


@patch(f"{MODULE_PATH}.get_llm_client")
def test_extract_facts_list_llm_enabled_success(mock_get_client, target_module):
    """Test successful JSON extraction from the LLM."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Simulate the LLM returning a valid JSON string
    mock_response.choices[0].message.content = '["Fact 1", "Fact 2"]'
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    with patch.dict(target_module.config, {"use_llm_for_facts": True}):
        result = target_module.extract_facts_list("Some text", "test.txt")

        assert result == ["Fact 1", "Fact 2"]
        mock_client.chat.completions.create.assert_called_once()


@patch(f"{MODULE_PATH}.get_llm_client")
def test_extract_facts_list_llm_markdown_stripping(mock_get_client, target_module):
    """Test that markdown code blocks are correctly stripped from the LLM response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Simulate the LLM wrapping the JSON in markdown blocks
    mock_response.choices[0].message.content = '```json\n["Fact from markdown"]\n```'

    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    with patch.dict(target_module.config, {"use_llm_for_facts": True}):
        result = target_module.extract_facts_list("Some text", "test.txt")

        assert result == ["Fact from markdown"]


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.get_llm_client")
def test_extract_facts_list_json_error(mock_get_client, mock_logger, target_module):
    """Test handling of invalid JSON returned by the LLM."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is not JSON at all."
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    with patch.dict(target_module.config, {"use_llm_for_facts": True}):
        result = target_module.extract_facts_list("Some text", "test.txt")

        assert result == []
        mock_logger.error.assert_called_once()
        assert "Error processing: test.txt" in mock_logger.error.call_args[0][0]


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.get_llm_client")
def test_extract_facts_list_api_exception(mock_get_client, mock_logger, target_module):
    """Test handling of an API connection error."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Timeout")
    mock_get_client.return_value = mock_client

    with patch.dict(target_module.config, {"use_llm_for_facts": True}):
        result = target_module.extract_facts_list("Some text", "test.txt")

        assert result == []
        mock_logger.error.assert_called_once()
        assert (
            "Error API for test.txt: API Timeout" in mock_logger.error.call_args[0][0]
        )


## Tests for main() loop


@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}.os.makedirs")
@patch(f"{MODULE_PATH}.logger")
def test_main_missing_input_folder(
    mock_logger, mock_makedirs, mock_exists, target_module
):
    """Test that main() gracefully skips missing folders."""
    # Force os.path.exists to return False for the input folders
    mock_exists.return_value = False

    target_module.main()

    mock_makedirs.assert_called_once_with(target_module.OUTPUT_DIR, exist_ok=True)
    # Should warn for both default folders
    assert mock_logger.warning.call_count == 2


@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}.is_facts_extracted")
@patch(f"{MODULE_PATH}.logger")
def test_main_skips_already_extracted(
    mock_logger, mock_is_extracted, mock_exists, mock_listdir, target_module
):
    """Test that main() skips files that are already marked as extracted."""
    # Simulate the folder exists, but the metadata files don't
    mock_exists.side_effect = lambda path: path in [
        "src/data/scraped_raw",
        "src/data/processed_text",
    ]
    mock_listdir.return_value = ["already_done.txt"]
    mock_is_extracted.return_value = True

    # Use a dummy mock_open to prevent FileNotFoundError when it tries to read the txt file
    with patch("builtins.open", mock_open(read_data="Dummy content")):
        target_module.main()

    # Verify it logged the skip
    mock_logger.info.assert_any_call("SKIP (already extracted): already_done.txt")
