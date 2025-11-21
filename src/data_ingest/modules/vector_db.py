from datetime import datetime

import chromadb


def save_to_vector_db(text_chunk, embedding, source_url, path_to_database):
    """
    Saves texts chunks, embeddings urls to vector database.

    Args:
        text_chunk (list[str] | str): Cleaned documents scrapped from mini website.
        embedding (list[list[float]] | list[float]): Document's embeddings
        source_url (list[str] | str): mini website urls
        path_to_database (str): path to local database

    Returns:
        Null
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
            ids=[f"ids_{number_of_docs+i+j+1}" for j in range(len(batch_texts))],
        )


def load_vector_db(path_to_database):
    """
    Returns vector database in given path.

    Args:

        path_to_database (str): path to local database

    Returns:
        vector database
    """

    # for use chroma locally, once we got docker set switch PersistentClient() -> HttpClient()
    chroma_client = chromadb.PersistentClient(path=path_to_database)

    collection = chroma_client.get_collection(name="mini_docs")

    return collection
