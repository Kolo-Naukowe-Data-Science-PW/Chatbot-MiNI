from unittest.mock import MagicMock, patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.api.query_rewriter"
# ==============================================================================
MODULE_PATH = "api.query_rewriter"


@pytest.fixture
def target_module():
    """Dynamically imports the target module and resets the global client state."""
    import importlib

    module = importlib.import_module(MODULE_PATH)

    # CRITICAL: Reset the global singleton before every test to ensure isolation
    module._client = None

    return module


# --- Tests for _get_client() ---


def test_get_client_missing_key(target_module, monkeypatch):
    """Test that a RuntimeError is raised if the API key is missing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        target_module._get_client()


@patch(f"{MODULE_PATH}.OpenAI")
def test_get_client_success_and_caching(mock_openai_class, target_module, monkeypatch):
    """Test that the client is initialized correctly and cached globally."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-key")

    # First call - should initialize the client
    client1 = target_module._get_client()

    # Second call - should return the exact same cached instance
    client2 = target_module._get_client()

    # Assert initialization happened exactly once with correct parameters
    mock_openai_class.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1", api_key="test-secret-key"
    )

    # Assert caching works
    assert client1 is client2
    assert client1 == mock_openai_class.return_value


# --- Tests for rewrite_query() ---


@patch(f"{MODULE_PATH}._get_client")
def test_rewrite_query_success(mock_get_client, target_module):
    """Test that the query is rewritten and stripped correctly on a successful API call."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Simulate LLM returning a string with extra whitespace
    mock_response.choices[0].message.content = "  egzamin sesja termin WMiNI  "
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = target_module.rewrite_query("Kiedy mam egzaminy w sesji?")

    assert result == "egzamin sesja termin WMiNI"
    mock_client.chat.completions.create.assert_called_once_with(
        model=target_module.REWRITE_MODEL,
        messages=[
            {"role": "system", "content": target_module.SYSTEM_PROMPT},
            {"role": "user", "content": "Kiedy mam egzaminy w sesji?"},
        ],
        temperature=0.0,
        max_tokens=60,
    )


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}._get_client")
def test_rewrite_query_empty_result(mock_get_client, mock_logger, target_module):
    """Test that it falls back to the original query if the LLM returns an empty string."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Simulate LLM returning only whitespace or None
    mock_response.choices[0].message.content = "   "
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    original_query = "Gdzie jest dziekanat?"
    result = target_module.rewrite_query(original_query)

    assert result == original_query
    mock_logger.warning.assert_called_once()
    assert "returned empty string" in mock_logger.warning.call_args[0][0]


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}._get_client")
def test_rewrite_query_exception_fallback(mock_get_client, mock_logger, target_module):
    """Test that it falls back to the original query if the API throws an exception."""
    mock_client = MagicMock()
    # Simulate API timeout or failure
    mock_client.chat.completions.create.side_effect = Exception("OpenRouter Timeout")
    mock_get_client.return_value = mock_client

    original_query = "Kto jest dziekanem?"
    result = target_module.rewrite_query(original_query)

    assert result == original_query
    mock_logger.warning.assert_called_once()
    assert "Query rewriting failed" in mock_logger.warning.call_args[0][0]
