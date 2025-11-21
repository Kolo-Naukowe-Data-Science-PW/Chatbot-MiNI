import logging
import os

from langchain_community.document_loaders import (
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)


def get_loader_for_file(file_path: str):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return UnstructuredPDFLoader(file_path, mode="elements", strategy="fast")

    elif ext in [".docx", ".doc"]:
        return UnstructuredWordDocumentLoader(file_path)

    elif ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(file_path)

    else:
        logging.warning(f"Unsupported file type: {ext}")
        return None
