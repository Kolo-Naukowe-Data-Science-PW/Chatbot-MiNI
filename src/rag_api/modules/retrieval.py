import logging
from data_ingest.modules.embedder import Embedder
from data_ingest.modules.vector_db import load_vector_db
from utils.paths import get_data_dir

logger = logging.getLogger(__name__)

DATABASE_PATH = get_data_dir("temp_database")

logger.info("Loading Embedder model for retrieval...")
embedder = Embedder()
logger.info("Embedder loaded.")


def get_top_k_chunks(query: str, top_k: int = 5):
    """
    Retrieves the top-k most relevant text chunks from the vector database based on the query.
    Each chunk is a dictionary with 'text_chunk' and 'source_url'.
    """
    logger.info("Starting retrieval for top %d chunks. Query: '%s'", top_k, query)

    try:
        logger.debug("Loading vector database from: %s", DATABASE_PATH) # <-- DODANE
        vector_db = load_vector_db(DATABASE_PATH)

        logger.debug("Generating embedding for query...")
        query_embedding = embedder.generate_embedding(query)

        logger.debug("Querying vector database...")
        top_chunks = vector_db.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        logger.info("Successfully retrieved results from vector_db.")
        return top_chunks

    except Exception as e:
        logger.error(
            "Failed during chunk retrieval: %s", e, exc_info=True
        )
        return {"documents": [[]], "metadatas": [[]], "ids": [[]]}