import json
import logging
import os

from data_ingest.modules.embedder import Embedder
from data_ingest.modules.vector_db import save_to_vector_db
from pipeline.common import CURRENT_VERSION
from utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "src/data/facts"
DB_PATH = os.environ.get("CHROMA_DIR", get_data_dir("chroma_db"))


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
