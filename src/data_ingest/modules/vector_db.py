"""
Heads-up:
- Chroma is currently set up to run locally; need to move it to the cloud later.
- Embeddings and hyperlinks of documents aren't added yet.
"""

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

    # handle the case when instead of list of values user provide single text chunk
    if type(text_chunk) is not list:
        text_chunk = [text_chunk]

    if type(embedding[0]) is not list:
        embedding = [embedding]

    if type(source_url) is not list:
        source_url = [source_url]

    # for use chroma locally, once we got docker set switch PersistentClient() -> HttpClient()
    chroma_client = chromadb.PersistentClient(path=path_to_database)

    # get_or_create_collection to avoid creating a new collection every time
    collection = chroma_client.get_or_create_collection(
        name="mini_docs",
        configuration={
            "hnsw": {  # if set chroma on cloud, probably need to change 'hnsw' -> 'spann'
                "space": "cosine"
            }
        },
        metadata={  # collection metadata
            "description": "Database with docs scrapped from mini website",
            "created": str(datetime.now()),
        },
    )

    # number of current docs in collection
    number_of_docs = collection.count()

    # add docs to collection
    collection.add(
        documents=text_chunk,
        embeddings=embedding,
        metadatas=[{"url": url} for url in source_url],
        ids=[f"ids_{number_of_docs+i+1}" for i in range(len(text_chunk))],
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
