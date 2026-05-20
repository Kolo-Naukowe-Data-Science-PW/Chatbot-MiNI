from unittest.mock import patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file containing Embedder!
# Example: "src.embeddings.embedder"
# ==============================================================================
MODULE_PATH = "ingestion.embedder"


@pytest.fixture
def mock_hf_embeddings():
    """Fixture to mock the LangChain HuggingFaceEmbeddings class."""
    with patch(f"{MODULE_PATH}.HuggingFaceEmbeddings") as mock_class:
        yield mock_class


def test_embedder_initialization_default(mock_hf_embeddings):
    """Test that Embedder initializes with the default model."""
    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    # Instantiate without arguments
    embedder = target_module.Embedder()

    assert embedder.model_name == "BAAI/bge-m3"
    mock_hf_embeddings.assert_called_once_with(model_name="BAAI/bge-m3")
    assert embedder.embedder == mock_hf_embeddings.return_value


def test_embedder_initialization_custom(mock_hf_embeddings):
    """Test that Embedder initializes with a custom model name."""
    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    custom_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = target_module.Embedder(model_name=custom_model)

    assert embedder.model_name == custom_model
    mock_hf_embeddings.assert_called_once_with(model_name=custom_model)


def test_generate_embedding(mock_hf_embeddings):
    """Test generating a single embedding delegates correctly."""
    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    # Setup mock return value
    mock_instance = mock_hf_embeddings.return_value
    expected_vector = [0.1, 0.2, 0.3, 0.4]
    mock_instance.embed_query.return_value = expected_vector

    embedder = target_module.Embedder()
    result = embedder.generate_embedding("Test text")

    # Assertions
    mock_instance.embed_query.assert_called_once_with("Test text")
    assert result == expected_vector


def test_generate_embeddings(mock_hf_embeddings):
    """Test generating multiple embeddings delegates correctly."""
    import importlib

    target_module = importlib.import_module(MODULE_PATH)

    # Setup mock return value
    mock_instance = mock_hf_embeddings.return_value
    expected_vectors = [[0.1, 0.2], [0.3, 0.4]]
    mock_instance.embed_documents.return_value = expected_vectors

    embedder = target_module.Embedder()
    test_texts = ["First document", "Second document"]
    result = embedder.generate_embeddings(test_texts)

    # Assertions
    mock_instance.embed_documents.assert_called_once_with(test_texts)
    assert result == expected_vectors
