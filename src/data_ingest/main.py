import json
import logging
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_ingest.modules.embedder import Embedder
from data_ingest.modules.extractor import get_loader_for_file
from data_ingest.modules.vector_db import save_to_vector_db
from scraper.main import STORAGE_DIR

logging.basicConfig(level=logging.INFO)

DATABASE_PATH = "temp_database"


def clean_text_structure(text: str) -> str:
    """
    Cleans text while preserving important layout structures like
    lists and paragraph breaks.
    """
    if not text:
        return ""

    text = re.sub(r"\n\s*\n", "\n\n", text)

    lines = text.split("\n")
    cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]

    return "\n".join(cleaned_lines).strip()


def ingest_documents():
    embedder = Embedder()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
    )

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
            split_docs = text_splitter.split_documents(raw_docs)

            doc_text_contents = []

            for doc in split_docs:
                cleaned_content = clean_text_structure(doc.page_content)
                if cleaned_content:
                    doc_text_contents.append(cleaned_content)

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
