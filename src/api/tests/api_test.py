# import json
# import pytest
# from fastapi.testclient import TestClient
# from unittest.mock import patch, MagicMock

## ==============================================================================
## IMPORTANT: Change this to the actual path of the file you are testing!
## Example: "src.api.routes"
## ==============================================================================
# MODULE_PATH = "api.api"


# @pytest.fixture
# def target_module():
#    """Dynamically imports the target module."""
#    import importlib
#    return importlib.import_module(MODULE_PATH)


# @pytest.fixture
# def client(target_module):
#    """Provides a FastAPI TestClient for the application."""
#    return TestClient(target_module.app)


## --- Tests for /experiment-config ---

# @patch("os.getenv")
# def test_experiment_config(mock_getenv, client):
#    """Test that the experiment config reads environment variables correctly."""
#    def env_side_effect(key, default):
#        mapping = {
#            "EXPERIMENT_PERSONA": "1",
#            "EXPERIMENT_TEMP": "0.5",
#            "EXPERIMENT_DIM": "temperature",
#            "EXPERIMENT_MODEL": "openai/gpt-4o",
#        }
#        return mapping.get(key, default)

#    mock_getenv.side_effect = env_side_effect

#    response = client.get("/experiment-config")
#    assert response.status_code == 200

#    data = response.json()
#    assert data["dim"] == "temperature"
#    assert data["baseline_model"] == "openai/gpt-4o"
#    assert data["baseline_temp"] == 0.5
#    assert data["persona_idx"] == 1
#    assert data["baseline_persona"] == "Odpowiedz luzno, prosto i przyjaźnie."


## --- Tests for /retrieval ---

# @patch(f"{MODULE_PATH}.get_top_k_chunks")
# @patch(f"{MODULE_PATH}.rewrite_query")
# def test_retrieval_endpoint(mock_rewrite, mock_get_chunks, client):
#    """Test the direct retrieval endpoint (ablation experiment)."""
#    mock_rewrite.return_value = "Rewritten Query"
#    mock_get_chunks.return_value = [
#        {"text_chunk": "Chunk 1", "source_url": "http://url1.com"},
#        {"text_chunk": "Chunk 2", "source_url": "http://url1.com"}, # Duplicate URL
#        {"text_chunk": "Chunk 3", "source_url": "http://url2.com"}
#    ]

#    payload = {
#        "query": "Original Query",
#        "top_k": 5,
#        "use_rerank": True,
#        "use_rewrite": True
#    }

#    response = client.post("/retrieval", json=payload)
#    assert response.status_code == 200

#    data = response.json()
#    assert data["retrieval_query"] == "Rewritten Query"
#    # Should deduplicate URLs
#    assert data["urls"] == ["http://url1.com", "http://url2.com"]

#    mock_rewrite.assert_called_once_with("Original Query")
#    mock_get_chunks.assert_called_once_with("Rewritten Query", top_k=5, use_rerank=True)


## --- Tests for /db-urls ---

# @patch(f"{MODULE_PATH}._get_qdrant_client")
# def test_db_urls_endpoint(mock_get_client, client):
#    """Test retrieving all unique URLs from the Qdrant DB."""
#    mock_qdrant = MagicMock()

#    # Simulate pagination: First call returns points + offset, second returns points + None
#    point1 = MagicMock(payload={"url": "http://url-a.com"})
#    point2 = MagicMock(payload={"url": "http://url-b.com"})
#    point3 = MagicMock(payload={"url": "http://url-a.com"}) # Duplicate

#    mock_qdrant.scroll.side_effect = [
#        ([point1, point2], "next_offset_token"),
#        ([point3], None)
#    ]
#    mock_get_client.return_value = mock_qdrant

#    response = client.get("/db-urls")
#    assert response.status_code == 200

#    data = response.json()
#    assert data["count"] == 2
#    assert data["urls"] == ["http://url-a.com", "http://url-b.com"]


## --- Tests for /errors and /feedback ---

# @patch(f"{MODULE_PATH}.get_data_dir")
# def test_error_report_endpoint(mock_get_data_dir, client, tmp_path):
#    """Test saving error reports to the filesystem."""
#    # Redirect the error directory to a pytest temporary path
#    mock_get_data_dir.return_value = str(tmp_path / "errors")

#    payload = {
#        "error_description": "Something broke",
#        "response_text": "Bad LLM response"
#    }

#    response = client.post("/errors", json=payload)
#    assert response.status_code == 200
#    assert response.json() == {"status": "ok"}

#    # Verify the file was created and contains the correct data
#    error_files = list((tmp_path / "errors").glob("*.json"))
#    assert len(error_files) == 1

#    with open(error_files[0], "r") as f:
#        saved_data = json.load(f)
#        assert saved_data["error_description"] == "Something broke"
#        assert "created_at" in saved_data


# def test_feedback_endpoint(client, target_module, tmp_path, monkeypatch):
#    """Test saving feedback to the CSV file."""
#    # Redirect the CSV file to a pytest temporary path
#    temp_csv = tmp_path / "model_feedback.csv"
#    monkeypatch.setattr(target_module, "feedback_file_path", temp_csv)

#    payload = {
#        "query": "Test query",
#        "rating": 5,
#        "variant_config": {"model": "gpt-4"}
#    }

#    response = client.post("/feedback", json=payload)
#    assert response.status_code == 200

#    # Verify the CSV was created and written to
#    assert temp_csv.exists()
#    content = temp_csv.read_text(encoding="utf-8")
#    assert "query" in content
#    assert "Test query" in content
#    assert "gpt-4" in content


## --- Tests for /chat ---

# @patch(f"{MODULE_PATH}.query_llm")
# @patch(f"{MODULE_PATH}.build_messages")
# @patch(f"{MODULE_PATH}.get_top_k_chunks")
# @patch(f"{MODULE_PATH}.rewrite_query")
# def test_chat_happy_path_pl(mock_rewrite, mock_get_chunks, mock_build, mock_query, client):
#    """Test the standard chat endpoint with Polish language (no translation needed)."""
#    mock_rewrite.return_value = "Rewritten query"
#    mock_get_chunks.return_value = [{"text_chunk": "Fact 1", "source_url": "http://src.com"}]
#    mock_build.return_value = [{"role": "user", "content": "..."}]
#    mock_query.return_value = "Odpowiedź LLM."

#    payload = {"query": "Jakie są zasady?", "language": "pl"}

#    response = client.post("/chat", json=payload)
#    assert response.status_code == 200

#    data = response.json()
#    assert data["answer"] == "Odpowiedź LLM."
#    assert data["sources"] == ["http://src.com"]
#    assert data["retrieval_query"] == "Rewritten query"

#    mock_rewrite.assert_called_once_with("Jakie są zasady?")


# @patch(f"{MODULE_PATH}.translate_text")
# @patch(f"{MODULE_PATH}.query_llm")
# @patch(f"{MODULE_PATH}.get_top_k_chunks")
# @patch(f"{MODULE_PATH}.rewrite_query")
# def test_chat_translation_flow(mock_rewrite, mock_get_chunks, mock_query, mock_translate, client):
#    """Test the chat endpoint when the user queries in a foreign language."""
#    mock_translate.side_effect = ["Przetłumaczone z EN na PL", "Translated back to EN"]
#    mock_rewrite.return_value = "Rewritten"
#    mock_get_chunks.return_value = [{"text_chunk": "Fact", "source_url": "url"}]
#    mock_query.return_value = "Polska odpowiedź"

#    payload = {"query": "What are the rules?", "language": "en"}

#    response = client.post("/chat", json=payload)
#    assert response.status_code == 200

#    data = response.json()
#    assert data["answer"] == "Translated back to EN"

#    # Verify translation was called twice (Incoming to PL, Outgoing to Target)
#    assert mock_translate.call_count == 2
#    mock_translate.assert_any_call("What are the rules?", target_lang_code="pl")
#    mock_translate.assert_any_call("Polska odpowiedź", target_lang_code="en")


# @patch(f"{MODULE_PATH}.translate_text")
# @patch(f"{MODULE_PATH}.get_top_k_chunks")
# @patch(f"{MODULE_PATH}.rewrite_query")
# def test_chat_no_context_found(mock_rewrite, mock_get_chunks, mock_translate, client):
#    """Test the fallback response when Qdrant returns 0 chunks."""
#    mock_rewrite.return_value = "Rewritten"
#    mock_get_chunks.return_value = [] # Empty DB results
#    mock_translate.return_value = "Sorry, I don't know."

#    payload = {"query": "Unknown topic", "language": "en"}

#    response = client.post("/chat", json=payload)
#    assert response.status_code == 200

#    data = response.json()
#    assert data["sources"] == []
#    assert data["answer"] == "Sorry, I don't know."


## --- Tests for /chat/stream ---

# @patch(f"{MODULE_PATH}.query_llm_stream")
# @patch(f"{MODULE_PATH}.get_top_k_chunks")
# @patch(f"{MODULE_PATH}.rewrite_query")
# def test_chat_stream_pl(mock_rewrite, mock_get_chunks, mock_query_stream, client):
#    """Test Server-Sent Events streaming in Polish."""
#    mock_rewrite.return_value = "Rewritten"
#    mock_get_chunks.return_value = [{"text_chunk": "Fact", "source_url": "url.com"}]

#    # Simulate a generator yielding tokens
#    def dummy_stream(*args, **kwargs):
#        yield "To "
#        yield "jest "
#        yield "strumień."

#    mock_query_stream.side_effect = dummy_stream

#    payload = {"query": "Test?", "language": "pl"}

#    with client.stream("POST", "/chat/stream", json=payload) as response:
#        assert response.status_code == 200

#        # Read the raw SSE response string
#        content = response.read().decode("utf-8")

#        assert "data: To \n\n" in content
#        assert "data: jest \n\n" in content
#        assert "data: strumień.\n\n" in content

#        # Verify the final JSON payload was sent
#        assert '{"event": "done", "sources": ["url.com"], "retrieval_query": "Rewritten"}' in content
