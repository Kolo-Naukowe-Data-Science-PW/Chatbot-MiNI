import json
import logging
import os

from src.data_ingest.modules.embedder import Embedder
from src.data_ingest.modules.vector_db import reset_collection, save_to_vector_db
from src.pipeline.common import CURRENT_VERSION
from src.utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "src/data/facts"
DB_PATH = os.environ.get("QDRANT_DIR", get_data_dir("qdrant_db"))


def main() -> None:
    """
    Ingests facts from JSON files, generates embeddings, and saves them to ChromaDB.

    This function reads all JSON files from the configured input directory, extracts
    facts and source URLs, generates vector embeddings for each fact, and stores
    everything in the vector database. It also logs progress and any errors encountered.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logger.info(f"Starting ingestion for pipeline version: {CURRENT_VERSION}")

    embedder = Embedder()

    all_text_chunks = []
    all_urls = []

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".json"))
    total_files = len(files)
    logger.info(f"Found {total_files} files with facts to ingest.")

    for i, filename in enumerate(files, start=1):
        path = os.path.join(INPUT_DIR, filename)

        try:
            with open(path, encoding="utf-8") as f:
                facts_list = json.load(f)

            if not isinstance(facts_list, list):
                logger.warning(
                    f"[{i}/{total_files}] {filename}: wrong format, expected a list of facts."
                )
                continue

            facts_in_file = [item for item in facts_list if item.get("fact")]
            logger.info(f"[{i}/{total_files}] {filename}: {len(facts_in_file)} facts")

            for item in facts_in_file:
                all_text_chunks.append(item["fact"])
                all_urls.append(item.get("source", "unknown"))

        except Exception as e:
            logger.error(f"[{i}/{total_files}] {filename}: error reading — {e}")

    if not all_text_chunks:
        logger.warning("No data to ingest.")
        return

    logger.info(f"Total facts to ingest: {len(all_text_chunks)}")
    logger.info(f"Resetting Qdrant collection before ingestion...")
    reset_collection(DB_PATH)

    logger.info(f"Generating embeddings for {len(all_text_chunks)} facts...")
    embeddings = embedder.generate_embeddings(all_text_chunks)

    logger.info(f"Saving to Qdrant ({DB_PATH})...")
    save_to_vector_db(all_text_chunks, embeddings, all_urls, DB_PATH)
    logger.info("Ingestion complete. Ready for deployment!")


if __name__ == "__main__":
    main()
