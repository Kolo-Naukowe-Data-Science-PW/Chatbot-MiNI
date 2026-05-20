import sys
from unittest.mock import MagicMock, patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.ingestion.vector_db"
# ==============================================================================
MODULE_PATH = "ingestion.vector_db"


@pytest.fixture
def target_module():
    """
    Dynamically imports the module while mocking fastembed to prevent
    the BM25 model from downloading/loading during test collection.
    """
    with patch("fastembed.SparseTextEmbedding"):
        import importlib

        # If the module was already loaded by another test, reload it while patched
        if MODULE_PATH in sys.modules:
            importlib.reload(sys.modules[MODULE_PATH])
        else:
            importlib.import_module(MODULE_PATH)

        return sys.modules[MODULE_PATH]


# --- Tests for Collection Management ---


def test_ensure_collection_creates_new(target_module):
    """Test that it creates a collection if it does not exist."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False

    target_module._ensure_collection(mock_client)

    mock_client.collection_exists.assert_called_once_with(target_module.COLLECTION_NAME)
    mock_client.create_collection.assert_called_once()

    # Verify it passed the right params (dense and sparse)
    kwargs = mock_client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == target_module.COLLECTION_NAME
    assert "dense" in kwargs["vectors_config"]
    assert "sparse" in kwargs["sparse_vectors_config"]


def test_ensure_collection_skips_existing(target_module):
    """Test that it skips creation if the collection already exists."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    target_module._ensure_collection(mock_client)

    mock_client.create_collection.assert_not_called()


@patch(f"{MODULE_PATH}._get_client")
@patch(f"{MODULE_PATH}._ensure_collection")
def test_reset_collection_existing(mock_ensure, mock_get_client, target_module):
    """Test resetting when the collection currently exists."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_get_client.return_value = mock_client

    target_module.reset_collection("dummy/path")

    mock_client.delete_collection.assert_called_once_with(target_module.COLLECTION_NAME)
    mock_ensure.assert_called_once_with(mock_client)


@patch(f"{MODULE_PATH}._get_client")
@patch(f"{MODULE_PATH}._ensure_collection")
def test_reset_collection_new(mock_ensure, mock_get_client, target_module):
    """Test resetting when the collection does NOT exist yet."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_get_client.return_value = mock_client

    target_module.reset_collection("dummy/path")

    mock_client.delete_collection.assert_not_called()
    mock_ensure.assert_called_once_with(mock_client)


# --- Tests for Sparse Computation ---


def test_compute_sparse(target_module):
    """Test that _compute_sparse correctly formats the output from fastembed."""
    # Setup the mock embedding object that fastembed normally yields
    mock_embedding = MagicMock()
    mock_embedding.indices.tolist.return_value = [1, 5, 9]
    mock_embedding.values.tolist.return_value = [0.5, 0.8, 0.1]

    # Configure our module's _sparse_model to return this mock data
    target_module._sparse_model.embed.return_value = [mock_embedding]

    result = target_module._compute_sparse(["test text"])

    assert len(result) == 1
    # Check that it packed them into a Qdrant SparseVector object correctly
    assert result[0].indices == [1, 5, 9]
    assert result[0].values == [0.5, 0.8, 0.1]


# --- Tests for Database Operations ---


@patch(f"{MODULE_PATH}._get_client")
@patch(f"{MODULE_PATH}._ensure_collection")
@patch(f"{MODULE_PATH}._compute_sparse")
def test_save_to_vector_db_single_item(
    mock_compute_sparse, mock_ensure, mock_get_client, target_module
):
    """Test saving a single string/embedding (verifies list normalization)."""
    mock_client = MagicMock()
    mock_client.count.return_value.count = 50  # Simulate 50 existing points
    mock_get_client.return_value = mock_client

    # FIX: Use a real SparseVector so Pydantic doesn't crush the MagicMock into []
    real_sparse_vector = target_module.SparseVector(indices=[1, 2], values=[0.5, 0.8])
    mock_compute_sparse.return_value = [real_sparse_vector]

    target_module.save_to_vector_db(
        text_chunk="Single fact",
        embedding=[0.1, 0.2, 0.3],
        source_url="http://source.com",
        path_to_database="dummy/path",
    )

    mock_client.upsert.assert_called_once()

    # Extract the points passed to upsert
    upsert_kwargs = mock_client.upsert.call_args.kwargs
    points = upsert_kwargs["points"]

    assert len(points) == 1
    point = points[0]

    assert point.id == 50  # 50 existing + 0 index
    assert point.payload["text"] == "Single fact"
    assert point.payload["url"] == "http://source.com"
    assert "created" in point.payload
    assert point.vector["dense"] == [0.1, 0.2, 0.3]

    # Check that Pydantic kept our actual SparseVector intact
    assert point.vector["sparse"].indices == [1, 2]
    assert point.vector["sparse"].values == [0.5, 0.8]


@patch(f"{MODULE_PATH}._get_client")
@patch(f"{MODULE_PATH}._ensure_collection")
@patch(f"{MODULE_PATH}._compute_sparse")
def test_save_to_vector_db_batching(
    mock_compute_sparse, mock_ensure, mock_get_client, target_module
):
    """Test that large inputs are chunked into multiple upsert calls."""
    mock_client = MagicMock()
    mock_client.count.return_value.count = 0
    mock_get_client.return_value = mock_client

    total_items = 6000
    texts = [f"Text {i}" for i in range(total_items)]
    embeddings = [[0.1, 0.2] for _ in range(total_items)]
    urls = [f"http://url{i}.com" for i in range(total_items)]

    # FIX: Use real SparseVectors
    real_sparse_vector = target_module.SparseVector(indices=[1], values=[0.1])
    mock_compute_sparse.return_value = [real_sparse_vector for _ in range(total_items)]

    target_module.save_to_vector_db(texts, embeddings, urls, "dummy/path")

    # Should be called twice (Batch 1: 5000, Batch 2: 1000)
    assert mock_client.upsert.call_count == 2

    # Check sizes of the batches
    first_call_points = mock_client.upsert.call_args_list[0].kwargs["points"]
    second_call_points = mock_client.upsert.call_args_list[1].kwargs["points"]

    assert len(first_call_points) == 5000
    assert len(second_call_points) == 1000


@patch(f"{MODULE_PATH}._get_client")
def test_load_vector_db(mock_get_client, target_module):
    """Test that load simply delegates to the client generator."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    result = target_module.load_vector_db("dummy/path")

    assert result == mock_client
    mock_get_client.assert_called_once_with("dummy/path")
