import numpy as np
from data_ingest.modules.embedder import Embedder
from data_ingest.modules.vector_db import load_vector_db
from rag_api.modules.config import VECTOR_DB_PATH

embedder = Embedder()


def cosine_similarity(vec1, vec2):
    """
    Calculates cosine similarity between two vectors.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_top_k_chunks(query: str, top_k: int = 5):
    """
    Retrieves the top-k most relevant text chunks from the vector database based on the query.
    Each chunks is a dictionary with 'text_chunk' and 'source_url'.
    """

    vector_db = load_vector_db(VECTOR_DB_PATH)

    query_embedding = embedder.generate_embedding(query)

    for chunk in vector_db:
        chunk['similarity'] = cosine_similarity(query_embedding, chunk['embedding'])

    sorted_chunks = sorted(vector_db, key=lambda x: x['similarity'], reverse=True)

    top_chunks = [{'text_chunk': c['text_chunk'], 'source_url': c['source_url']} 
                  for c in sorted_chunks[:top_k]]

    return top_chunks