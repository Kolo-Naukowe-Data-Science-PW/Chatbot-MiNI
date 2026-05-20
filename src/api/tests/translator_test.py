from unittest.mock import MagicMock, patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.api.translator"
# ==============================================================================
MODULE_PATH = "api.translator"


@pytest.fixture
def target_module():
    """Dynamically imports the target module."""
    import importlib

    return importlib.import_module(MODULE_PATH)


def test_translate_text_empty_input(target_module):
    """Test that empty or None text returns an empty string immediately."""
    assert target_module.translate_text("", "en") == ""
    assert target_module.translate_text(None, "pl") == ""


@patch(f"{MODULE_PATH}.get_llm_client")
def test_translate_text_success_english(mock_get_client, target_module):
    """Test successful translation to English."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Simulate API returning text with extra spaces to verify stripping
    mock_response.choices[0].message.content = "   Translated to English.   "
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    original_text = "Przetłumacz to na angielski."
    result = target_module.translate_text(original_text, "en")

    # Verify the text was stripped and returned
    assert result == "Translated to English."

    # Verify the LLM was called with correct parameters
    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args.kwargs

    assert kwargs["model"] == target_module.MODEL_WORKER
    assert kwargs["temperature"] == 0.1

    # Verify the system prompt included the correct language name
    system_prompt = kwargs["messages"][0]["content"]
    assert "Translate the following text into English." in system_prompt

    # Verify the user message contained the exact input text
    user_message = kwargs["messages"][1]["content"]
    assert user_message == original_text


@patch(f"{MODULE_PATH}.get_llm_client")
def test_translate_text_unsupported_language_fallback(mock_get_client, target_module):
    """Test that an unknown language code defaults to Polish."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Polski tekst"
    )
    mock_get_client.return_value = mock_client

    target_module.translate_text("Some text", "fr")  # "fr" is not in the lang_map

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_prompt = kwargs["messages"][0]["content"]

    # It should fall back to "Polish"
    assert "Translate the following text into Polish." in system_prompt


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}.get_llm_client")
def test_translate_text_exception_fallback(mock_get_client, mock_logger, target_module):
    """Test that an API failure logs an error and returns the original text."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API is down")
    mock_get_client.return_value = mock_client

    original_text = "To jest ważny tekst."
    result = target_module.translate_text(original_text, "ua")

    # It should safely return the original text
    assert result == original_text

    # Verify the error was logged correctly
    mock_logger.error.assert_called_once()
    assert (
        "Translation to Ukrainian failed: API is down"
        in mock_logger.error.call_args[0][0]
    )
