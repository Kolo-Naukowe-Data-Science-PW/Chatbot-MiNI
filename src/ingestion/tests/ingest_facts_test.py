import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.ingestion.ingest_facts"
# ==============================================================================
MODULE_PATH = "ingestion.ingest_facts"


@pytest.fixture
def target_module():
    """Dynamically imports the target module to avoid early-load environment errors."""
    import importlib

    return importlib.import_module(MODULE_PATH)


# --- Tests for _load_facts_from_file ---


def test_load_facts_success(target_module):
    """Test successful loading of texts and URLs from a valid JSON file."""
    mock_json = [
        {"fact": "Fact 1", "source": "http://url1.com"},
        {"fact": "Fact 2", "source": "http://url2.com"},
        {"fact": "Fact 3"},  # Missing source should fallback to 'unknown'
    ]

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_json))):
        texts, urls = target_module._load_facts_from_file(
            "dummy_path.json", "dummy_path.json"
        )

    assert texts == ["Fact 1", "Fact 2", "Fact 3"]
    assert urls == ["http://url1.com", "http://url2.com", "unknown"]


def test_load_facts_not_a_list(target_module, caplog):
    """Test handling of a JSON file that contains a dict instead of a list."""
    mock_json = {"fact": "This is a single dict, not a list"}

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_json))):
        texts, urls = target_module._load_facts_from_file(
            "bad_format.json", "bad_format.json"
        )

    assert texts == []
    assert urls == []
    assert "expected a list" in caplog.text


def test_load_facts_exception(target_module, caplog):
    """Test handling of a malformed JSON file."""
    with patch("builtins.open", mock_open(read_data="[Not valid JSON}")):
        texts, urls = target_module._load_facts_from_file("broken.json", "broken.json")

    assert texts == []
    assert urls == []
    assert "error reading" in caplog.text


# --- Tests for main() ---


@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}.logger")
def test_main_input_dir_missing(mock_logger, mock_exists, target_module):
    """Test that main() aborts if the input directory doesn't exist."""
    mock_exists.return_value = False

    target_module.main()

    mock_logger.error.assert_called_once()
    assert "Input directory does not exist" in mock_logger.error.call_args[0][0]


@patch(f"{MODULE_PATH}.is_ingested")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}.logger")
def test_main_no_pending_files(
    mock_logger, mock_exists, mock_listdir, mock_is_ingested, target_module
):
    """Test that main() stops early if all files are already ingested."""
    mock_exists.return_value = True
    mock_listdir.return_value = ["file1.json", "file2.json"]
    mock_is_ingested.return_value = True  # Both files are marked as ingested

    target_module.main()

    mock_logger.info.assert_any_call("Nothing to ingest — all files already processed.")


@patch(f"{MODULE_PATH}.mark_ingested")
@patch(f"{MODULE_PATH}.save_to_vector_db")
@patch(f"{MODULE_PATH}.Embedder")
@patch(f"{MODULE_PATH}.reset_collection")
@patch(f"{MODULE_PATH}.ingested_count")
@patch(f"{MODULE_PATH}.is_ingested")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}._load_facts_from_file")
def test_main_fresh_start_and_batching(
    mock_load_facts,
    mock_exists,
    mock_listdir,
    mock_is_ingested,
    mock_count,
    mock_reset,
    mock_embedder_class,
    mock_save,
    mock_mark,
    target_module,
):
    """Test a full run: fresh start (resets DB) and proper batching behavior."""
    mock_exists.return_value = True
    # Simulate 3 files
    mock_listdir.return_value = ["f1.json", "f2.json", "f3.json"]
    mock_is_ingested.return_value = False

    # 0 ingested count means fresh start -> triggers reset_collection
    mock_count.return_value = 0

    # Setup the mock embedder
    mock_embedder = MagicMock()
    mock_embedder.generate_embeddings.return_value = [
        [0.1, 0.2]
    ]  # Dummy embedding vector
    mock_embedder_class.return_value = mock_embedder

    # Mock file loading to return dummy text/urls
    mock_load_facts.return_value = (["A fact"], ["http://url"])

    # Overwrite BATCH_SIZE to 2 to test the batching loop with 3 files (should create 2 batches)
    with patch.object(target_module, "BATCH_SIZE", 2):
        target_module.main()

    # Assertions
    mock_reset.assert_called_once_with(target_module.DB_PATH)

    # 3 files with batch size 2 = 2 batches = 2 saves to the DB
    assert mock_save.call_count == 2
    assert mock_embedder.generate_embeddings.call_count == 2

    # mark_ingested should be called for every processed file (3 times total)
    assert mock_mark.call_count == 3


@patch(f"{MODULE_PATH}.mark_ingested")
@patch(f"{MODULE_PATH}.save_to_vector_db")
@patch(f"{MODULE_PATH}.Embedder")
@patch(f"{MODULE_PATH}.reset_collection")
@patch(f"{MODULE_PATH}.ingested_count")
@patch(f"{MODULE_PATH}.is_ingested")
@patch(f"{MODULE_PATH}.os.listdir")
@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}._load_facts_from_file")
def test_main_resume_ingestion(
    mock_load_facts,
    mock_exists,
    mock_listdir,
    mock_is_ingested,
    mock_count,
    mock_reset,
    mock_embedder_class,
    mock_save,
    mock_mark,
    target_module,
):
    """Test that resuming ingestion does NOT wipe the existing Qdrant collection."""
    mock_exists.return_value = True
    mock_listdir.return_value = ["f1.json"]
    mock_is_ingested.return_value = False

    # Count > 0 means we are resuming -> do NOT reset collection
    mock_count.return_value = 5
    mock_load_facts.return_value = (["Fact"], ["http://url"])

    target_module.main()

    mock_reset.assert_not_called()
    mock_save.assert_called_once()
