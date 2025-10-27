"""
Utility functions for path resolution across the project.
"""

import os


def find_repo_root():
    """
    Find the repository root by searching upward for pyproject.toml.
    Returns:
        str: Absolute path to the repository root directory.
    Raises:
        RuntimeError: If pyproject.toml cannot be found.
    """

    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)
    raise RuntimeError("Could not find repository root (no pyproject.toml found)")


def get_data_dir(*subdirs):
    """
    Get a path within the src/data directory.
    Args:
        *subdirs: Subdirectories within src/data to join.
    Returns:
        str: Absolute path to the specified data directory.
    Example:
        >>> get_data_dir("raw", "scraper")
        '/path/to/repo/src/data/raw/scraper'
    """
    
    repo_root = find_repo_root()
    return os.path.join(repo_root, "src", "data", *subdirs)