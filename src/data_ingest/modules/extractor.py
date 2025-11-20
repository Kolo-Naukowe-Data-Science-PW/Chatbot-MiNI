import logging
import os

from langchain.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)


def get_loader_for_file(file_path: str):
    """
    Return an appropriate LangChain document loader for the given file path.

    The loader is selected based on the file extension:
      - .pdf  -> PyPDFLoader
      - .docx, .doc -> UnstructuredWordDocumentLoader
      - .html, .htm -> BSHTMLLoader (opened with UTF-8 encoding)

    Args:
        file_path: Absolute or relative path to the file on disk.

    Returns:
        A loader instance ready to call `.load()` on success, or None if
        the file type is unsupported (a warning is logged).
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        return UnstructuredWordDocumentLoader(file_path)
    elif ext in [".html", ".htm"]:
        return BSHTMLLoader(file_path, open_encoding="utf-8")
    else:
        logging.warning(f"Unsupported file type: {ext}")
        return None
