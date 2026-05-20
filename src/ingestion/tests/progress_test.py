import json

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.ingestion.progress"
# ==============================================================================
MODULE_PATH = "ingestion.progress"


@pytest.fixture
def target_module(tmp_path, monkeypatch):
    """
    Dynamically imports the target module and redirects PROGRESS_FILE
    to a temporary directory so real files are never touched.
    """
    import importlib

    module = importlib.import_module(MODULE_PATH)

    # Create a path inside the pytest temporary directory
    temp_file = tmp_path / "test_progress.json"

    # Monkeypatch the module's PROGRESS_FILE variable to point to our temp file
    monkeypatch.setattr(module, "PROGRESS_FILE", str(temp_file))

    return module


# --- Tests for _load and _save ---


def test_load_missing_file(target_module):
    """Test that a missing file returns the empty skeleton dict."""
    data = target_module._load()
    assert data == {"scraped": [], "facts_extracted": [], "ingested": []}


def test_load_corrupt_file(target_module, caplog):
    """Test that an invalid JSON file falls back to an empty dict and warns."""
    # Write garbage to the temp file
    with open(target_module.PROGRESS_FILE, "w") as f:
        f.write("{ Not valid JSON")

    data = target_module._load()

    assert data == {"scraped": [], "facts_extracted": [], "ingested": []}
    assert "Could not read progress file" in caplog.text


def test_load_missing_keys(target_module):
    """Test that _load gracefully fills in missing keys from older file versions."""
    incomplete_data = {"scraped": ["http://test.com"]}

    with open(target_module.PROGRESS_FILE, "w") as f:
        json.dump(incomplete_data, f)

    data = target_module._load()

    assert data["scraped"] == ["http://test.com"]
    # The missing keys should be automatically added
    assert data["facts_extracted"] == []
    assert data["ingested"] == []


def test_save_creates_directories_and_file(target_module):
    """Test that _save properly handles atomic writes and folder creation."""
    test_data = {
        "scraped": ["url1"],
        "facts_extracted": ["file1"],
        "ingested": ["file1"],
    }

    # Prove the file doesn't exist yet
    import os

    assert not os.path.exists(target_module.PROGRESS_FILE)

    target_module._save(test_data)

    # Prove the file now exists and contains the exact data
    assert os.path.exists(target_module.PROGRESS_FILE)
    with open(target_module.PROGRESS_FILE) as f:
        saved_data = json.load(f)

    assert saved_data == test_data


# --- Tests for Read Functions ---


def test_is_functions(target_module):
    """Test the boolean check functions."""
    target_module._save(
        {"scraped": ["url_a"], "facts_extracted": ["file_b"], "ingested": ["file_c"]}
    )

    assert target_module.is_scraped("url_a") is True
    assert target_module.is_scraped("url_x") is False

    assert target_module.is_facts_extracted("file_b") is True
    assert target_module.is_facts_extracted("file_x") is False

    assert target_module.is_ingested("file_c") is True
    assert target_module.is_ingested("file_x") is False


def test_aggregate_functions(target_module):
    """Test ingested_count and get_status."""
    target_module._save(
        {
            "scraped": ["url1", "url2", "url3"],
            "facts_extracted": ["file1", "file2"],
            "ingested": ["file1"],
        }
    )

    assert target_module.ingested_count() == 1

    status = target_module.get_status()
    assert status == {"scraped": 3, "facts_extracted": 2, "ingested": 1}


# --- Tests for Write Functions ---


def test_mark_functions_add_and_prevent_duplicates(target_module):
    """Test that marking items saves them, but ignores duplicates."""

    # 1. Mark items for the first time
    target_module.mark_scraped("url1")
    target_module.mark_facts_extracted("file1")
    target_module.mark_ingested("file1")

    status = target_module.get_status()
    assert status["scraped"] == 1
    assert status["facts_extracted"] == 1
    assert status["ingested"] == 1

    # 2. Mark the EXACT SAME items a second time
    target_module.mark_scraped("url1")
    target_module.mark_facts_extracted("file1")
    target_module.mark_ingested("file1")

    # 3. Prove that duplicates were ignored (counts remain 1)
    status_after_dup = target_module.get_status()
    assert status_after_dup["scraped"] == 1
    assert status_after_dup["facts_extracted"] == 1
    assert status_after_dup["ingested"] == 1

    # 4. Mark NEW items
    target_module.mark_scraped("url2")
    assert target_module.get_status()["scraped"] == 2
