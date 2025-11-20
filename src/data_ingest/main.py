import json
import logging
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter

from data_ingest.modules.embedder import Embedder
from data_ingest.modules.extractor import get_loader_for_file
from data_ingest.modules.vector_db import save_to_vector_db
from scraper.main import OUTPUT_DIR
from utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)
DATABASE_PATH = get_data_dir("temp_database")


def ingest_documents():
    """
    1. Load documents (PDF/DOCX/HTML) using LangChain loaders.
    2. Read companion JSON files for URL metadata.
    3. Split text into chunks using LangChain Splitter.
    4. Embed chunks.
    5. Save to Vector DB.
    """

    embedder = Embedder()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    all_files = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
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
        else:
            logging.warning(f"No metadata JSON found for {file_path}")

        try:
            raw_docs = loader.load()

            split_docs = text_splitter.split_documents(raw_docs)

            doc_text_contents = []

            for doc in split_docs:
                content = " ".join(doc.page_content.split())

                if content:
                    doc_text_contents.append(content)

            if not doc_text_contents:
                logging.warning(f"No text extracted from {file_path}")
                continue

            embeddings_for_doc = embedder.generate_embeddings(doc_text_contents)

            all_chunks.extend(doc_text_contents)
            all_embeddings.extend(embeddings_for_doc)
            all_urls.extend([source_url] * len(doc_text_contents))

        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")

    if all_chunks:
        save_to_vector_db(all_chunks, all_embeddings, all_urls, DATABASE_PATH)
        logging.info(f"Ingestion complete. Saved {len(all_chunks)} chunks.")
    else:
        logging.warning("No chunks were generated. Database update skipped.")


if __name__ == "__main__":
    ingest_documents()
