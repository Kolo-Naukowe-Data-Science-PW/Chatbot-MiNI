"""
Schedule-only ingest pipeline.

Replaces previously ingested schedule (plany zajęć) data with freshly
scraped and extracted facts. Run this instead of the full scrape+ingest
when only schedule pages need updating.

Steps:
  1. Scrape USOS schedule pages (direct HTTP + BeautifulSoup).
  2. Delete old schedule facts from Qdrant (matched by source URL).
  3. Clear progress markers for schedule files.
  4. Extract facts with schedule-specialized LLM prompt.
  5. Embed and ingest new schedule facts into Qdrant using UUID point IDs.
"""

import json
import logging
import os

from src.ingestion.embedder import Embedder
from src.ingestion.extract_schedule_facts import main as run_extraction
from src.ingestion.links_extended import links
from src.ingestion.progress import clear_progress_for_prefix, mark_ingested
from src.ingestion.scrape_schedule import SCHEDULE_PATTERN, scrape_schedules
from src.ingestion.vector_db import delete_by_url_list, save_to_vector_db_uuid
from src.utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACTS_DIR = "src/data/facts"
DB_PATH = os.environ.get("QDRANT_DIR", get_data_dir("qdrant_db"))


def _get_schedule_urls() -> list[str]:
    return [u for u in links if SCHEDULE_PATTERN in u]


def _ingest_schedule_facts() -> int:
    """Embed and ingest all schedule_*_facts.json files into Qdrant."""
    fact_files = sorted(
        f
        for f in os.listdir(FACTS_DIR)
        if f.startswith("schedule_") and f.endswith("_facts.json")
    )

    if not fact_files:
        logger.warning("No schedule fact files found to ingest.")
        return 0

    logger.info(f"Ingesting {len(fact_files)} schedule fact files...")
    embedder = Embedder()
    total_facts = 0

    for filename in fact_files:
        path = os.path.join(FACTS_DIR, filename)
        try:
            with open(path, encoding="utf-8") as f:
                facts_list = json.load(f)
            items = [item for item in facts_list if item.get("fact")]
            if not items:
                logger.warning(f"No facts in {filename}")
                continue
            texts = [item["fact"] for item in items]
            urls = [item.get("source", "unknown") for item in items]

            embeddings = embedder.generate_embeddings(texts)
            save_to_vector_db_uuid(texts, embeddings, urls, DB_PATH)
            mark_ingested(filename)
            total_facts += len(texts)
            logger.info(f"  Ingested {len(texts)} facts from {filename}")
        except Exception as e:
            logger.error(f"Error ingesting {filename}: {e}")

    return total_facts


def main() -> None:
    schedule_urls = _get_schedule_urls()
    logger.info(f"=== SCHEDULE PIPELINE — {len(schedule_urls)} schedule URLs ===")

    logger.info("--- STEP 1: Scraping schedule pages ---")
    scrape_schedules()

    logger.info("--- STEP 2: Deleting old schedule facts from Qdrant ---")
    delete_by_url_list(schedule_urls, DB_PATH)
    logger.info(f"  Deleted points for {len(schedule_urls)} URLs.")

    logger.info("--- STEP 3: Clearing progress markers for schedule files ---")
    clear_progress_for_prefix("schedule_", stages=["facts_extracted", "ingested"])

    logger.info("--- STEP 4: Extracting schedule facts ---")
    run_extraction()

    logger.info("--- STEP 5: Ingesting schedule facts into Qdrant ---")
    n = _ingest_schedule_facts()
    logger.info(f"  Ingested {n} total schedule facts.")

    logger.info("=== SCHEDULE PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
