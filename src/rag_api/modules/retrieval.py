from data_ingest.modules.embedder import Embedder
from data_ingest.modules.vector_db import load_vector_db
from utils.paths import get_data_dir

DATABASE_PATH = get_data_dir("temp_database")
embedder = Embedder()


def get_top_k_chunks(query: str, top_k: int = 5):
    """
    Retrieves the top-k most relevant text chunks from the vector database based on the query.
    Each chunks is a dictionary with 'text_chunk' and 'source_url'.
    """

    vector_db = load_vector_db(DATABASE_PATH)

    query_embedding = embedder.generate_embedding(query)

    top_chunks = vector_db.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    return top_chunks
