import json
import logging
import os
import sys

from common import CURRENT_VERSION

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_ingest.modules.embedder import Embedder
from data_ingest.modules.vector_db import save_to_vector_db
from utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "new_pipeline/data/facts"
DB_PATH = os.environ.get("CHROMA_DIR", get_data_dir("chroma_db"))


def main():
    """
    Main function to ingest facts from JSON files, generate embeddings, and save them to ChromaDB.
    1. Reads all JSON files from the INPUT_DIR.
    2. Extracts fact text and source URL from each entry.
    3. Generates embeddings for the fact texts.
    4. Saves the texts, embeddings, and URLs to ChromaDB.
    5. Logs progress and any errors encountered.
    """

    logger.info(f"Starting ingestion for pipeline version: {CURRENT_VERSION}")

    embedder = Embedder()

    all_text_chunks = []
    all_urls = []

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    logger.info(f"Found {len(files)} files with facts to ingest.")

    for filename in files:

        path = os.path.join(INPUT_DIR, filename)

        try:

            with open(path, encoding="utf-8") as f:
                facts_list = json.load(f)

            if not isinstance(facts_list, list):
                logger.warning(
                    f"File {filename} has wrong format, expected a list of facts."
                )
                continue

            for item in facts_list:
                fact_text = item.get("fact")
                source_url = item.get("source", "unknown")
                if fact_text:
                    all_text_chunks.append(fact_text)
                    all_urls.append(source_url)

        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")

    if not all_text_chunks:
        logger.warning("No data to ingest.")
        return

    logger.info(f"Generating embeddings for {len(all_text_chunks)} facts...")
    embeddings = embedder.generate_embeddings(all_text_chunks)

    logger.info(f"Saving to ChromaDB ({DB_PATH})...")
    save_to_vector_db(all_text_chunks, embeddings, all_urls, DB_PATH)
    logger.info("Ready for deployment!")


if __name__ == "__main__":
    main()
