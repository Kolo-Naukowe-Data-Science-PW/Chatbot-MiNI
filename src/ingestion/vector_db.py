import uuid
from datetime import datetime

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

COLLECTION_NAME = "mini_docs"
DENSE_DIM = 1024  # BAAI/bge-m3


def _get_client(path_to_database: str) -> QdrantClient:
    return QdrantClient(path=path_to_database)


def _ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )


def reset_collection(path_to_database: str) -> None:
    client = _get_client(path_to_database)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    _ensure_collection(client)


def _compute_sparse(texts: list[str]) -> list[SparseVector]:
    results = []
    for embedding in _sparse_model.embed(texts):
        indices = embedding.indices.tolist()
        values = embedding.values.tolist()
        results.append(SparseVector(indices=indices, values=values))
    return results


def save_to_vector_db(
    text_chunk: str | list[str],
    embedding: list[float] | list[list[float]],
    source_url: str | list[str],
    path_to_database: str,
) -> None:
    if not isinstance(text_chunk, list):
        text_chunk = [text_chunk]
    if not isinstance(embedding[0], list):
        embedding = [embedding]
    if not isinstance(source_url, list):
        source_url = [source_url]

    client = _get_client(path_to_database)
    _ensure_collection(client)

    # Oblicz sparse embeddings (BM25)
    sparse_vectors = _compute_sparse(text_chunk)

    # Pobierz aktualną liczbę punktów jako offset ID
    count = client.count(COLLECTION_NAME).count

    batch_size = 5000
    for i in range(0, len(text_chunk), batch_size):
        batch_texts = text_chunk[i : i + batch_size]
        batch_dense = embedding[i : i + batch_size]
        batch_sparse = sparse_vectors[i : i + batch_size]
        batch_urls = source_url[i : i + batch_size]

        points = [
            PointStruct(
                id=count + i + j,
                vector={
                    "dense": batch_dense[j],
                    "sparse": batch_sparse[j],
                },
                payload={
                    "text": batch_texts[j],
                    "url": batch_urls[j],
                    "created": str(datetime.now()),
                },
            )
            for j in range(len(batch_texts))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)


def delete_by_url_list(urls: list[str], path_to_database: str) -> None:
    """Delete all Qdrant points whose 'url' payload matches any of the given URLs."""
    if not urls:
        return
    client = _get_client(path_to_database)
    if not client.collection_exists(COLLECTION_NAME):
        return
    chunk_size = 100
    for i in range(0, len(urls), chunk_size):
        batch = urls[i : i + chunk_size]
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="url", match=MatchAny(any=batch))]
                )
            ),
        )


def save_to_vector_db_uuid(
    text_chunk: str | list[str],
    embedding: list[float] | list[list[float]],
    source_url: str | list[str],
    path_to_database: str,
) -> None:
    """Like save_to_vector_db but uses UUID point IDs — safe to call after deletions."""
    if not isinstance(text_chunk, list):
        text_chunk = [text_chunk]
    if not isinstance(embedding[0], list):
        embedding = [embedding]
    if not isinstance(source_url, list):
        source_url = [source_url]

    client = _get_client(path_to_database)
    _ensure_collection(client)
    sparse_vectors = _compute_sparse(text_chunk)

    batch_size = 5000
    for i in range(0, len(text_chunk), batch_size):
        batch_texts = text_chunk[i : i + batch_size]
        batch_dense = embedding[i : i + batch_size]
        batch_sparse = sparse_vectors[i : i + batch_size]
        batch_urls = source_url[i : i + batch_size]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": batch_dense[j], "sparse": batch_sparse[j]},
                payload={
                    "text": batch_texts[j],
                    "url": batch_urls[j],
                    "created": str(datetime.now()),
                },
            )
            for j in range(len(batch_texts))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)


def load_vector_db(path_to_database: str) -> QdrantClient:
    return _get_client(path_to_database)


# ---------------------------------------------------------------------------
# mini_chunks collection helpers
# ---------------------------------------------------------------------------

COLLECTION_NAME_CHUNKS = "mini_chunks"


def _ensure_chunks_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME_CHUNKS):
        client.create_collection(
            collection_name=COLLECTION_NAME_CHUNKS,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )


def ensure_chunks_collection(path_to_database: str) -> None:
    """Create mini_chunks Qdrant collection if it doesn't exist."""
    client = _get_client(path_to_database)
    _ensure_chunks_collection(client)


def save_chunks_to_vector_db(
    text_chunk: str | list[str],
    embedding: list[float] | list[list[float]],
    source_url: str | list[str],
    path_to_database: str,
) -> None:
    """Save text chunks to the mini_chunks collection using UUID point IDs."""
    if not isinstance(text_chunk, list):
        text_chunk = [text_chunk]
    if not isinstance(embedding[0], list):
        embedding = [embedding]
    if not isinstance(source_url, list):
        source_url = [source_url]

    client = _get_client(path_to_database)
    _ensure_chunks_collection(client)
    sparse_vectors = _compute_sparse(text_chunk)

    batch_size = 5000
    for i in range(0, len(text_chunk), batch_size):
        batch_texts = text_chunk[i : i + batch_size]
        batch_dense = embedding[i : i + batch_size]
        batch_sparse = sparse_vectors[i : i + batch_size]
        batch_urls = source_url[i : i + batch_size]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": batch_dense[j], "sparse": batch_sparse[j]},
                payload={
                    "text": batch_texts[j],
                    "url": batch_urls[j],
                    "created": str(datetime.now()),
                },
            )
            for j in range(len(batch_texts))
        ]
        client.upsert(collection_name=COLLECTION_NAME_CHUNKS, points=points)
