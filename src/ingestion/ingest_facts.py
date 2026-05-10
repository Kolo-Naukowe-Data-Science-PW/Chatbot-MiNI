import json
import logging
import os

from src.ingestion.embedder import Embedder
from src.ingestion.vector_db import reset_collection, save_to_vector_db
from src.ingestion.common import CURRENT_VERSION
from src.ingestion.progress import ingested_count, is_ingested, mark_ingested
from src.utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "src/data/facts"
DB_PATH = os.environ.get("QDRANT_DIR", get_data_dir("qdrant_db"))

BATCH_SIZE = 10


def _load_facts_from_file(path: str, filename: str) -> tuple[list[str], list[str]]:
    """Return (texts, urls) for a single facts JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            facts_list = json.load(f)
        if not isinstance(facts_list, list):
            logger.warning(f"{filename}: expected a list, got {type(facts_list).__name__}")
            return [], []
        facts = [item for item in facts_list if item.get("fact")]
        texts = [item["fact"] for item in facts]
        urls = [item.get("source", "unknown") for item in facts]
        return texts, urls
    except Exception as e:
        logger.error(f"{filename}: error reading — {e}")
        return [], []


def main() -> None:
    """
    Ingest facts into Qdrant in batches of BATCH_SIZE files.

    - Skips files already recorded in the progress tracker.
    - Resets the Qdrant collection only on a fresh start (nothing ingested yet).
    - Safe to re-run after interruption: picks up where it left off.

    To start completely fresh: delete src/data/pipeline_progress.json
    and src/data/qdrant_db/.
    """
    logger.info(f"Starting ingestion for pipeline version: {CURRENT_VERSION}")

    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory does not exist: {INPUT_DIR}")
        return

    all_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".json"))
    pending = [f for f in all_files if not is_ingested(f)]

    already_done = len(all_files) - len(pending)
    logger.info(
        f"Found {len(all_files)} fact files. "
        f"Already ingested: {already_done}. Pending: {len(pending)}."
    )

    if not pending:
        logger.info("Nothing to ingest — all files already processed.")
        return

    if ingested_count() == 0:
        logger.info("Fresh start: resetting Qdrant collection...")
        reset_collection(DB_PATH)
    else:
        logger.info(f"Resuming: {ingested_count()} files already in Qdrant, skipping reset.")

    embedder = Embedder()
    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, batch_start in enumerate(range(0, len(pending), BATCH_SIZE), start=1):
        batch_files = pending[batch_start : batch_start + BATCH_SIZE]
        batch_texts: list[str] = []
        batch_urls: list[str] = []
        processed_files: list[str] = []

        for filename in batch_files:
            path = os.path.join(INPUT_DIR, filename)
            texts, urls = _load_facts_from_file(path, filename)
            if texts:
                batch_texts.extend(texts)
                batch_urls.extend(urls)
                processed_files.append(filename)
            else:
                logger.warning(f"Skipping empty/broken file: {filename}")

        if not batch_texts:
            logger.warning(f"Batch {batch_num}/{total_batches}: no facts to ingest, skipping.")
            for filename in processed_files:
                mark_ingested(filename)
            continue

        logger.info(
            f"Batch {batch_num}/{total_batches}: "
            f"embedding {len(batch_texts)} facts from {len(processed_files)} files..."
        )
        embeddings = embedder.generate_embeddings(batch_texts)

        logger.info(f"Batch {batch_num}/{total_batches}: saving to Qdrant...")
        save_to_vector_db(batch_texts, embeddings, batch_urls, DB_PATH)

        for filename in processed_files:
            mark_ingested(filename)

        logger.info(
            f"Batch {batch_num}/{total_batches}: done. "
            f"Total ingested so far: {ingested_count()} files."
        )

    logger.info(f"Ingestion complete. Total ingested: {ingested_count()}/{len(all_files)} files.")


if __name__ == "__main__":
    main()
