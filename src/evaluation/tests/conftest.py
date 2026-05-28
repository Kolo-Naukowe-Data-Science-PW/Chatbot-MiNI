"""
Test configuration and fixtures for evaluation metrics tests.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# pytest fixtures for common test data


@pytest.fixture
def sample_text_pair():
    """Common text pair for quick tests."""
    return {
        "hypothesis": "the quick brown fox",
        "reference": "the quick brown fox jumps over the lazy dog",
    }


@pytest.fixture
def sample_relevance_scores():
    """Common relevance scores for retrieval metric tests."""
    return {
        "perfect": [1.0, 1.0, 1.0, 0.0],
        "mixed": [1.0, 0.0, 1.0, 0.0, 0.5],
        "reversed": [0.0, 0.0, 0.0, 1.0, 1.0],
        "empty": [],
    }


@pytest.fixture
def sample_urls():
    """Common URLs for hierarchical relevance tests."""
    return {
        "base": "https://example.com/path/to/page",
        "child": "https://example.com/path/to/page/child",
        "grandchild": "https://example.com/path/to/page/child/grandchild",
        "parent": "https://example.com/path/to",
        "different_domain": "https://other.com/path/to/page",
        "different_scheme": "http://example.com/path/to/page",
    }
