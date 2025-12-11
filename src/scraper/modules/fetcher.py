import json
import logging
import os
from urllib.parse import urlparse

import requests

logging.basicConfig(level=logging.INFO)


def sanitize_filename(url: str) -> str:
    """
    Sanitizes the URL to create a safe filename.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in url)


def fetch_file(url: str, output_dir: str, session: requests.Session = None) -> str:
    """
    Download a file from a URL, infer an appropriate file extension, and
    persist both the file and a companion metadata JSON in the given directory.

    The function determines the file extension using the HTTP Content-Type header
    and, as a fallback, the URL suffix. It writes the response body to disk and
    creates a JSON file alongside it containing basic provenance metadata.

    Args:
        url: The HTTP(S) URL to fetch.
        output_dir: Directory where the downloaded file and its metadata will be saved.
        session: Optional requests.Session object for connection reuse.

    Returns:
        The filesystem path to the saved file on success; None if the fetch
        or write fails (an error is logged in that case).
    """
    parsed = urlparse(url)
    clean_path = parsed.path.lower()
    if any(
        clean_path.endswith(ext)
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico"]
    ):
        logging.info(f"Skipping image URL: {url}")
        return None

    filename_check = sanitize_filename(url).lower()
    if any(
        ext in filename_check
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    ):

        if "_bwg_" in filename_check or "photo-gallery" in filename_check:
            logging.info(f"Skipping suspected gallery image: {url}")
            return None

    try:
        logging.info("Fetching URL: %s", url)
        if session:
            response = session.get(url, stream=True, timeout=20)
        else:
            response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()

        content_type = (response.headers.get("Content-Type") or "").lower()

        if content_type.startswith("image/"):
            logging.info(f"Skipping image content type: {content_type} for URL: {url}")
            return None

        filename_base = sanitize_filename(url)

        ext = ".html"
        if "application/pdf" in content_type:
            ext = ".pdf"
        elif "wordprocessingml" in content_type or "msword" in content_type:
            ext = ".docx"
        elif (
            "application/zip" in content_type
            or "application/x-zip-compressed" in content_type
        ):
            ext = ".zip"
        elif clean_path.endswith(".pdf"):
            ext = ".pdf"
        elif clean_path.endswith(".docx"):
            ext = ".docx"
        elif clean_path.endswith(".zip"):
            ext = ".zip"

        filename = filename_base + ext
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, filename)
        meta_path = os.path.join(output_dir, filename_base + ".json")

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        metadata = {
            "source_url": url,
            "content_type": content_type,
            "original_filename": filename,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Saved {filename} and metadata.")
        return file_path

    except Exception as e:
        logging.error("Failed to fetch %s: %s", url, e)
        return None