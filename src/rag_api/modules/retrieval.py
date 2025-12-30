import logging
import os
from typing import Any

from data_ingest.modules.embedder import Embedder
from data_ingest.modules.vector_db import load_vector_db
from utils.paths import get_data_dir

logger = logging.getLogger(__name__)

DATABASE_PATH = os.environ.get("CHROMA_DIR", get_data_dir("chroma_db"))

logger.info("Loading Embedder model for retrieval...")
embedder = Embedder()
logger.info("Embedder loaded.")


def get_top_k_chunks(
    query: str, top_k: int = 5
) -> list[dict[str, Any]]:  # previously top_k: int = 15
    """
    Retrieves the top-k most relevant text chunks.
    Returns a list of dicts: [{'text_chunk': str, 'source_url': str}, ...]
    """
    logger.info("Starting retrieval for top %d chunks. Query: '%s'", top_k, query)

    try:
        logger.debug("Loading vector database from: %s", DATABASE_PATH)
        vector_db = load_vector_db(DATABASE_PATH)

        logger.debug("Generating embedding for query...")
        query_embedding = embedder.generate_embeddings([query])

        logger.debug("Querying vector database...")
        results = vector_db.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        structured_results = []

        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]

            for doc, meta in zip(documents, metadatas, strict=False):
                structured_results.append(
                    {"text_chunk": doc, "source_url": meta.get("url", "Unknown Source")}
                )

        logger.info("Successfully retrieved %d results.", len(structured_results))
        return structured_results

    except Exception as e:
        logger.error("Failed during chunk retrieval: %s", e, exc_info=True)
        return []
