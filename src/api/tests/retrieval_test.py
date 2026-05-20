import sys
from unittest.mock import MagicMock, patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.api.retrieval"
# ==============================================================================
MODULE_PATH = "api.retrieval"


@pytest.fixture
def target_module():
    """
    Dynamically imports the module while mocking all heavy ML models to prevent
    downloads and slow initializations during test collection.
    """
    with (
        patch("src.ingestion.embedder.Embedder"),
        patch("fastembed.SparseTextEmbedding"),
        patch("sentence_transformers.CrossEncoder"),
    ):

        import importlib

        # If the module was already loaded, reload it under the context of our patches
        if MODULE_PATH in sys.modules:
            importlib.reload(sys.modules[MODULE_PATH])
        else:
            importlib.import_module(MODULE_PATH)

        module = sys.modules[MODULE_PATH]

        # Reset the global Qdrant client singleton before every test
        module._qdrant_client = None

        return module


# --- Tests for _get_qdrant_client ---


@patch(f"{MODULE_PATH}.load_vector_db")
def test_get_qdrant_client_singleton(mock_load, target_module):
    """Test that the Qdrant client is initialized only once and cached."""
    mock_load.return_value = "MockedClientInstance"

    # Call twice
    client1 = target_module._get_qdrant_client()
    client2 = target_module._get_qdrant_client()

    # Assert it returns the mock and caches it (singleton)
    assert client1 == "MockedClientInstance"
    assert client1 is client2

    # Ensure load_vector_db was only triggered once
    mock_load.assert_called_once_with(target_module.DATABASE_PATH)


# --- Tests for _get_sparse_vector ---


def test_get_sparse_vector(target_module):
    """Test that fastembed outputs are correctly mapped into Qdrant's SparseVector."""
    # Setup mock output from fastembed
    mock_fastembed_result = MagicMock()
    mock_fastembed_result.indices.tolist.return_value = [5, 10, 15]
    mock_fastembed_result.values.tolist.return_value = [0.1, 0.5, 0.9]

    target_module.sparse_model.embed.return_value = [mock_fastembed_result]

    result = target_module._get_sparse_vector("Test query")

    # Verify the mapping
    assert result.indices == [5, 10, 15]
    assert result.values == [0.1, 0.5, 0.9]
    target_module.sparse_model.embed.assert_called_once_with(["Test query"])


# --- Tests for get_top_k_chunks ---


@patch(f"{MODULE_PATH}._get_qdrant_client")
def test_get_top_k_chunks_empty_results(mock_get_client, target_module):
    """Test handling when Qdrant returns no results."""
    mock_client = MagicMock()
    # Return empty points
    mock_client.query_points.return_value.points = []
    mock_get_client.return_value = mock_client

    # Mock dense vectors so it doesn't crash before querying
    target_module.embedder.generate_embeddings.return_value = [[0.1, 0.2]]

    results = target_module.get_top_k_chunks("query")

    assert results == []


@patch(f"{MODULE_PATH}._get_sparse_vector")
@patch(f"{MODULE_PATH}._get_qdrant_client")
def test_get_top_k_chunks_no_rerank(mock_get_client, mock_get_sparse, target_module):
    """Test that setting use_rerank=False skips the cross-encoder entirely."""
    mock_client = MagicMock()
    mock_point = MagicMock()
    mock_point.payload = {"text": "Chunk 1", "url": "http://example.com"}
    mock_client.query_points.return_value.points = [mock_point, mock_point]
    mock_get_client.return_value = mock_client

    target_module.embedder.generate_embeddings.return_value = [[0.1]]
    mock_get_sparse.return_value = "mock_sparse_vector"

    # Run with reranking disabled, requesting top 1
    results = target_module.get_top_k_chunks("query", top_k=1, use_rerank=False)

    assert len(results) == 1
    assert results[0]["text_chunk"] == "Chunk 1"

    # Verify reranker was completely skipped
    target_module.reranker.predict.assert_not_called()


@patch(f"{MODULE_PATH}._get_sparse_vector")
@patch(f"{MODULE_PATH}._get_qdrant_client")
def test_get_top_k_chunks_with_rerank_and_deduplication(
    mock_get_client, mock_get_sparse, target_module
):
    """Test that the cross-encoder correctly scores and deduplicates URLs."""

    # Setup 3 candidate points. Notice points 1 and 2 share the same URL!
    p1 = MagicMock(payload={"text": "Bad chunk", "url": "http://site-A.com"})
    p2 = MagicMock(payload={"text": "Best chunk", "url": "http://site-A.com"})
    p3 = MagicMock(payload={"text": "Okay chunk", "url": "http://site-B.com"})

    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [p1, p2, p3]
    mock_get_client.return_value = mock_client

    target_module.embedder.generate_embeddings.return_value = [[0.1]]
    mock_get_sparse.return_value = "mock_sparse_vector"

    # Provide mock reranker scores mapping to p1, p2, and p3
    # p2 has the highest score for site-A, so p1 should be deduplicated away!
    target_module.reranker.predict.return_value = [0.1, 0.9, 0.5]

    results = target_module.get_top_k_chunks("Test query", top_k=10, use_rerank=True)

    # It should have filtered down to 2 unique URLs, sorted by highest score first
    assert len(results) == 2
    assert results[0]["text_chunk"] == "Best chunk"  # Score 0.9 (Site A)
    assert results[0]["source_url"] == "http://site-A.com"

    assert results[1]["text_chunk"] == "Okay chunk"  # Score 0.5 (Site B)
    assert results[1]["source_url"] == "http://site-B.com"

    # Verify predict was called with the correct text pairs
    target_module.reranker.predict.assert_called_once_with(
        [
            ("Test query", "Bad chunk"),
            ("Test query", "Best chunk"),
            ("Test query", "Okay chunk"),
        ]
    )


@patch(f"{MODULE_PATH}.logger")
@patch(f"{MODULE_PATH}._get_qdrant_client")
def test_get_top_k_chunks_exception_handling(
    mock_get_client, mock_logger, target_module
):
    """Test that a database failure is safely caught and returns an empty list."""
    mock_get_client.side_effect = Exception("Database is offline")

    results = target_module.get_top_k_chunks("query")

    assert results == []
    mock_logger.error.assert_called_once()
    assert "Failed during chunk retrieval" in mock_logger.error.call_args[0][0]
