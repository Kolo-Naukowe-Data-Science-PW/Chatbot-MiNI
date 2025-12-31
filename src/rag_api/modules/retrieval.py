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
) -> list[dict[str, Any]]:
    """
    Retrieves the top-k most relevant text chunks from the vector database.

    Connects to the ChromaDB, generates an embedding for the user query,
    and performs a similarity search to find the most relevant documents.

    Parameters
    ----------
    query : str
        The user's search query.
    top_k : int, optional
        The number of top results to retrieve, by default 5.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries, where each dictionary contains:
        - 'text_chunk': The text content of the retrieved document.
        - 'source_url': The URL source of the document.
    """
    logger.info("Starting retrieval for top %d chunks. Query: '%s'", top_k, query)

    try:
        logger.debug("Loading vector database from: %s", DATABASE_PATH)
        vector_db = load_vector_db(DATABASE_PATH)

        logger.debug("Generating embedding for query...")
        # Embedder expects a list of strings and returns a list of vectors
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
                    {
                        "text_chunk": doc,
                        "source_url": meta.get("url", "Unknown Source"),
                    }
                )

        logger.info("Successfully retrieved %d results.", len(structured_results))
        return structured_results

    except Exception as e:
        logger.error("Failed during chunk retrieval: %s", e, exc_info=True)
        return []