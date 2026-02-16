from datetime import datetime

import chromadb
from chromadb.api.models.Collection import Collection


def save_to_vector_db(
    text_chunk: str | list[str],
    embedding: list[float] | list[list[float]],
    source_url: str | list[str],
    path_to_database: str,
) -> None:
    """
    Saves text chunks, embeddings, and URLs to the vector database.

    Parameters
    ----------
    text_chunk : str | list[str]
        Cleaned documents scrapped from the MiNI website. Can be a single string or a list of strings.
    embedding : list[float] | list[list[float]]
        The embeddings corresponding to the documents. Can be a single vector or a list of vectors.
    source_url : str | list[str]
        The source URLs for the documents. Can be a single URL string or a list of strings.
    path_to_database : str
        The local file path to the persistent Chroma database.

    Returns
    -------
    None
    """
    if not isinstance(text_chunk, list):
        text_chunk = [text_chunk]

    if not isinstance(embedding[0], list):
        embedding = [embedding]

    if not isinstance(source_url, list):
        source_url = [source_url]

    settings = chromadb.config.Settings(anonymized_telemetry=False)
    chroma_client = chromadb.PersistentClient(path=path_to_database, settings=settings)

    collection = chroma_client.get_or_create_collection(
        name="mini_docs",
        metadata={
            "description": "Database with docs scrapped from mini website",
            "created": str(datetime.now()),
            "hnsw:space": "cosine",
        },
    )

    number_of_docs = collection.count()

    batch_size = 5000
    total_docs = len(text_chunk)

    for i in range(0, total_docs, batch_size):
        batch_texts = text_chunk[i : i + batch_size]
        batch_embeddings = embedding[i : i + batch_size]
        batch_urls = source_url[i : i + batch_size]

        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=[{"url": url} for url in batch_urls],
            ids=[f"ids_{number_of_docs + i + j + 1}" for j in range(len(batch_texts))],
        )


def load_vector_db(path_to_database: str) -> Collection:
    """
    Retrieves the vector database collection from the given path.

    Parameters
    ----------
    path_to_database : str
        The local file path to the persistent Chroma database.

    Returns
    -------
    chromadb.api.models.Collection.Collection
        The ChromaDB collection object named 'mini_docs'.
    """
    # for use chroma locally, once we got docker set switch PersistentClient() -> HttpClient()
    chroma_client = chromadb.PersistentClient(path=path_to_database)

    collection = chroma_client.get_collection(name="mini_docs")

    return collection
