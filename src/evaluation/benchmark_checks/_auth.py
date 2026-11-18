"""Authentication header for the technical API endpoints used by the checks."""

import os


def internal_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """
    Build the request headers for ``/retrieval`` and ``/db-urls``.

    The API protects these endpoints with a shared secret. The key is read from
    the ``INTERNAL_API_KEY`` environment variable (docker-compose passes it to
    the containers that run the checks).

    Parameters
    ----------
    extra : dict[str, str] | None
        Headers to keep, for example ``Content-Type``.

    Returns
    -------
    dict[str, str]
        ``extra`` plus ``X-Internal-Key`` when the key is configured.
    """
    headers = dict(extra or {})
    key = os.getenv("INTERNAL_API_KEY", "")
    if key:
        headers["X-Internal-Key"] = key
    return headers
