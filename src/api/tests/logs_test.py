import logging
from logging.handlers import RotatingFileHandler

import pytest

# ==============================================================================
# IMPORTANT: Change this to the actual path of the file you are testing!
# Example: "src.utils.logging_config"
# ==============================================================================
MODULE_PATH = "api.logs"


@pytest.fixture
def target_module():
    """Dynamically imports the target module."""
    import importlib

    return importlib.import_module(MODULE_PATH)


@pytest.fixture(autouse=True)
def preserve_root_logger():
    """
    CRITICAL: pytest uses the root logger. If we don't back up and restore
    its state, testing setup_logging() will permanently alter pytest's own
    output and break other test files!
    """
    root_logger = logging.getLogger()
    old_handlers = list(root_logger.handlers)
    old_level = root_logger.level

    yield  # Let the test run

    # Restore the root logger exactly how pytest had it
    root_logger.handlers = old_handlers
    root_logger.setLevel(old_level)


# --- Tests for _parse_log_level ---


def test_parse_log_level_valid_upper(target_module):
    """Test valid uppercase log levels."""
    assert target_module._parse_log_level("DEBUG") == logging.DEBUG
    assert target_module._parse_log_level("WARNING") == logging.WARNING
    assert target_module._parse_log_level("CRITICAL") == logging.CRITICAL


def test_parse_log_level_case_insensitive(target_module):
    """Test that lowercase and mixed case strings are parsed correctly."""
    assert target_module._parse_log_level("error") == logging.ERROR
    assert target_module._parse_log_level("InFo") == logging.INFO


def test_parse_log_level_invalid_fallback(target_module):
    """Test that invalid strings or empty strings fall back to INFO."""
    assert target_module._parse_log_level("SUPER_DEBUG") == logging.INFO
    assert target_module._parse_log_level("") == logging.INFO
    assert target_module._parse_log_level("123") == logging.INFO


# --- Tests for setup_logging ---


def test_setup_logging_console_only(target_module, monkeypatch):
    """Test setup_logging when no file is provided."""
    # Ensure no environment variable interference
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    target_module.setup_logging()

    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO

    # Should only have 1 handler (the StreamHandler to stderr)
    assert len(root_logger.handlers) == 1
    handler = root_logger.handlers[0]

    assert isinstance(handler, logging.StreamHandler)
    assert not isinstance(handler, RotatingFileHandler)
    # NOTSET on the handler means it defers to the root logger's level
    assert handler.level == logging.NOTSET


def test_setup_logging_with_file(target_module, tmp_path):
    """Test setup_logging creates a RotatingFileHandler when a file is provided."""
    log_file = tmp_path / "test_app.log"

    target_module.setup_logging(log_file=str(log_file))

    root_logger = logging.getLogger()

    # Should have 2 handlers: StreamHandler AND RotatingFileHandler
    assert len(root_logger.handlers) == 2

    # Extract the file handler
    file_handlers = [
        h for h in root_logger.handlers if isinstance(h, RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    fh = file_handlers[0]

    assert fh.baseFilename == str(log_file)
    assert fh.maxBytes == 5 * 1024 * 1024
    assert fh.backupCount == 5


def test_setup_logging_respects_env_var(target_module, monkeypatch):
    """Test that the LOG_LEVEL environment variable overrides the default level."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    target_module.setup_logging()

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG


def test_setup_logging_clears_existing_handlers(target_module):
    """Test that running setup_logging twice doesn't duplicate handlers."""
    root_logger = logging.getLogger()

    # Add dummy handlers to simulate an already-configured logger
    root_logger.addHandler(logging.NullHandler())
    root_logger.addHandler(logging.NullHandler())

    # This should clear the NullHandlers and add exactly 1 StreamHandler
    target_module.setup_logging()

    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], logging.StreamHandler)
