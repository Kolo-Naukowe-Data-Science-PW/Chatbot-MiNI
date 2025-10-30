import logging
import os

from data_ingest.modules.chunker import chunk_text
from data_ingest.modules.cleaner import clean_html
from data_ingest.modules.embedder import Embedder
from data_ingest.modules.extractor import extract_text
from data_ingest.modules.vector_db import save_to_vector_db
from scraper.main import STORAGE_DIR
from utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)

DATABASE_PATH = get_data_dir("temp_database")


def ingest_documents():
    """
    1. Load new documents from the scraper storage.
    2. Clean HTML files.
    3. Create text chunks.
    4. Embed the text chunks.
    5. Save embeddings, text chunks and URLs to the vector database.
    """

    embedder = Embedder()
    all_files = [os.path.join(STORAGE_DIR, f) for f in os.listdir(STORAGE_DIR)]
    logging.info("Found %d files to ingest.", len(all_files))

    embeddings = []
    chunks = []
    urls = []

    for file_path in all_files:

        logging.info("Processing file: %s", file_path)
        with open(file_path, encoding="utf-8") as f:
            url_line = f.readline().strip()
            raw_content = extract_text(f)

        cleaned_text = clean_html(raw_content)

        chunk = chunk_text(cleaned_text)

        embedding = embedder.generate_embeddings(chunks)

        chunks.append(chunk)
        embeddings.append(embedding)
        urls.append(url_line)

    save_to_vector_db(chunks, embeddings, urls, DATABASE_PATH)

    logging.info("Ingestion complete for all documents.")


if __name__ == "__main__":
    ingest_documents()
