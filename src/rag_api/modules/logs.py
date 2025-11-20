"""
Module for configuring standardized logging for the RAG API.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s"
DEFAULT_LOG_LEVEL = "INFO"


def _parse_log_level(value: str) -> int:
    """
    Safe parse of LOG_LEVEL names/numbers; fallback to INFO.

    Parameters
    ----------
    value : str
        The log level string to parse.

    Returns
    -------
    int
        The corresponding logging level constant.
    """
    try:
        # getLevelByName handles string names ("INFO")
        lvl = value.upper()
        numeric = logging.getLevelByName(lvl)
        if isinstance(numeric, int):
            return numeric
    except Exception:
        pass
    # fallback
    return logging.INFO


def setup_logging(log_file: str | None = None) -> None:
    """
    Configure the root logger for the application.

    Call once at application startup before other imports that create loggers.
    If log_file is provided, a rotating file handler will also be created.

    Parameters
    ----------
    log_file : Optional[str], default=None
        Path to a file where logs should be written. If None, logs
        will only go to the console (stderr).
    """
    log_level_env = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    level = _parse_log_level(log_level_env)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on re-run
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console handler (always active)
    console_handler = logging.StreamHandler(sys.stderr)
    # NOTSET makes the handler inherit the level from the root logger
    console_handler.setLevel(logging.NOTSET)
    console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    root_logger.addHandler(console_handler)

    # Optional file handler (with rotation)
    if log_file:
        # Rotate: 5MB per file, keep 5 backups
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info("Logging configured. Level set to %s.", logging.getLevelName(level))
    if log_file:
        logger.info("Logging to file: %s", log_file)
