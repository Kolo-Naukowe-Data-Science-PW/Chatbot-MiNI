import json
import logging
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_ingest.modules.embedder import Embedder
from data_ingest.modules.extractor import get_loader_for_file
from data_ingest.modules.vector_db import save_to_vector_db
from scraper.main import STORAGE_DIR
from utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)

DATABASE_PATH = os.getenv("CHROMA_DIR", get_data_dir("temp_database"))


def clean_text_structure(text: str) -> str:
    """Normalize whitespace while keeping paragraph breaks."""
    if not text:
        return ""
    text = re.sub(r"\n\s*\n", "\n\n", text)
    lines = text.split("\n")
    cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
    return "\n".join(cleaned_lines).strip()


def looks_like_enumerator(text: str) -> bool:
    """
    Detect lines that are likely just list / item markers, e.g.:
    '1)', '2.', 'a)', 'III.', 'Â§ 1.' etc.
    These should almost never be standalone chunks.
    """
    stripped = text.strip()

    if not stripped:
        return False

    if re.match(r"^(\d+[\.\)]|[a-zA-Z][\)\.]|[ivxlcdmIVXLCDM]+\.)$", stripped):
        return True

    if re.match(r"^Â§\s*\d+\.?$", stripped):
        return True

    return False


def merge_small_docs(
    docs,
    min_chunk_chars: int = 150,
    max_chunk_chars: int = 1500,
):
    """
    Merge very small chunks (including enumerators) into neighbors.

    - Any chunk shorter than `min_chunk_chars` OR that looks like
      a pure enumerator is considered "too small".
    - We try to merge with the previous chunk if possible; otherwise
      it becomes the start of a new buffer and will merge forward.
    """
    merged = []
    buffer_doc = None

    def is_too_small(d) -> bool:
        text = d.page_content or ""
        return len(text.strip()) < min_chunk_chars or looks_like_enumerator(text)

    for doc in docs:
        if buffer_doc is None:
            buffer_doc = doc
            continue

        buffer_text = buffer_doc.page_content or ""
        current_text = doc.page_content or ""

        if is_too_small(buffer_doc) or is_too_small(doc):
            combined_len = len(buffer_text) + 1 + len(current_text)
            if combined_len <= max_chunk_chars:
                buffer_doc.page_content = (
                    buffer_text.rstrip() + "\n" + current_text.lstrip()
                ).strip()
            else:
                merged.append(buffer_doc)
                buffer_doc = doc
        else:
            merged.append(buffer_doc)
            buffer_doc = doc

    if buffer_doc is not None:
        merged.append(buffer_doc)

    return merged


def split_documents_hybrid(
    documents,
    chunk_size: int = 500,
    overlap: int = 100,
    min_chunk_chars: int = 150,
    max_chunk_chars: int = 1500,
):
    """
    Hybrid splitter:
      1. Use RecursiveCharacterTextSplitter for initial splitting.
      2. Merge too-small / enumerator-like chunks with neighbors.
      3. Clean each chunk's text.
    """
    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    logging.info(
        "Initial splitting with chunk_size=%d, overlap=%d",
        chunk_size,
        overlap,
    )
    raw_chunks = base_splitter.split_documents(documents)

    logging.info("Got %d raw chunks; merging small onesâ€¦", len(raw_chunks))
    merged_chunks = merge_small_docs(
        raw_chunks,
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
    )

    final_chunks = []
    for doc in merged_chunks:
        cleaned = clean_text_structure(doc.page_content)
        if cleaned:
            doc.page_content = cleaned
            final_chunks.append(doc)

    logging.info(
        "After merging and cleaning: %d chunks (min=%d chars, max=%d chars)",
        len(final_chunks),
        min_chunk_chars,
        max_chunk_chars,
    )
    return final_chunks


def ingest_documents():
    embedder = Embedder()

    all_files = [
        os.path.join(STORAGE_DIR, f)
        for f in os.listdir(STORAGE_DIR)
        if not f.endswith(".json")
    ]
    logging.info("Found %d content files to ingest.", len(all_files))

    all_chunks: list[str] = []
    all_embeddings: list[list[float]] = []
    all_urls: list[str] = []

    for file_path in all_files:
        logging.info(f"Processing file: {file_path}")

        loader = get_loader_for_file(file_path)
        if not loader:
            continue

        base_name = os.path.splitext(file_path)[0]
        json_path = base_name + ".json"
        source_url = "unknown"

        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    meta_data = json.load(f)
                    source_url = meta_data.get("source_url", "unknown")
            except Exception as e:
                logging.warning(f"Could not read metadata for {file_path}: {e}")

        try:
            raw_docs = loader.load()

            split_docs = split_documents_hybrid(
                raw_docs,
                chunk_size=500,
                overlap=100,
                min_chunk_chars=150,
                max_chunk_chars=1500,
            )

            doc_text_contents = [doc.page_content for doc in split_docs]

            if not doc_text_contents:
                continue

            embeddings_for_doc = embedder.generate_embeddings(doc_text_contents)

            all_chunks.extend(doc_text_contents)
            all_embeddings.extend(embeddings_for_doc)
            all_urls.extend([source_url] * len(doc_text_contents))

        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")

    if all_chunks:
        total_chunks = len(all_chunks)
        logging.info(
            f"Total chunks to save: {total_chunks}. Saving to vector database..."
        )

        save_to_vector_db(all_chunks, all_embeddings, all_urls, DATABASE_PATH)

        logging.info(f"Ingestion complete. Saved {total_chunks} chunks.")
    else:
        logging.warning("No chunks were generated.")


if __name__ == "__main__":
    ingest_documents()
