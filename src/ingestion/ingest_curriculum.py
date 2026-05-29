"""
Curriculum-only ingest pipeline.

Replaces old plan-studiów facts in Qdrant with freshly extracted ones.
Run this (via 'Ingest Curriculum Only' workflow) when study programme
documents change. Does not touch schedule or other data.

Steps:
  1. Delete old curriculum facts from Qdrant (matched by exact PDF URLs).
  2. Clear progress markers for curriculum fact files.
  3. Extract facts with curriculum-specialized LLM prompt.
  4. Embed and ingest new curriculum facts using UUID point IDs.
"""

import json
import logging
import os

from src.ingestion.embedder import Embedder
from src.ingestion.extract_curriculum_facts import _is_curriculum_url
from src.ingestion.extract_curriculum_facts import main as run_extraction
from src.ingestion.links_extended import links
from src.ingestion.progress import clear_progress_for_prefix, mark_ingested
from src.ingestion.vector_db import delete_by_url_list, save_to_vector_db_uuid
from src.utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACTS_DIR = "src/data/facts"
DB_PATH = os.environ.get("QDRANT_DIR", get_data_dir("qdrant_db"))


def _get_curriculum_urls() -> list[str]:
    return [u for u in links if _is_curriculum_url(u)]


def _ingest_curriculum_facts() -> int:
    fact_files = sorted(
        f
        for f in os.listdir(FACTS_DIR)
        if f.startswith("curriculum_") and f.endswith("_facts.json")
    )

    if not fact_files:
        logger.warning("No curriculum fact files found to ingest.")
        return 0

    logger.info(f"Ingesting {len(fact_files)} curriculum fact files...")
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
    curriculum_urls = _get_curriculum_urls()
    logger.info(f"=== CURRICULUM PIPELINE — {len(curriculum_urls)} curriculum URLs ===")

    logger.info("--- STEP 1: Deleting old curriculum facts from Qdrant ---")
    delete_by_url_list(curriculum_urls, DB_PATH)
    logger.info(f"  Deleted points for {len(curriculum_urls)} URLs.")

    logger.info("--- STEP 2: Clearing progress markers for curriculum files ---")
    clear_progress_for_prefix("curriculum_", stages=["facts_extracted", "ingested"])

    logger.info("--- STEP 3: Extracting curriculum facts ---")
    run_extraction()

    logger.info("--- STEP 4: Ingesting curriculum facts into Qdrant ---")
    n = _ingest_curriculum_facts()
    logger.info(f"  Ingested {n} total curriculum facts.")

    logger.info("=== CURRICULUM PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
