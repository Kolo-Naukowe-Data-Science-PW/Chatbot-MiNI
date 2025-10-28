"""
This script loads clean .txt documents from a mini's website
and stores them in a Chroma collection for retrieval.

Heads-up:
- Chroma is currently set up to run locally; need to move it to the cloud later.
- Embeddings and hyperlinks of documents aren't added yet.
"""

import os
from datetime import datetime
from pathlib import Path

import chromadb

# get path to the directory containing cleaned text files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "cleaned")
INPUT_DIR = Path(os.path.abspath(INPUT_DIR))


def load_docs(path_to_docs) -> list:

    files = [f for f in path_to_docs.iterdir() if f.is_file()]

    # initialize list to hold content of docs
    docs = [None for _ in range(len(files))]

    # Read and store cleaned txt files in docs
    for idx, file in enumerate(files):
        with file.open("r", encoding="utf-8") as f:
            docs[idx] = f.read()

    return docs


def main():

    docs = load_docs(INPUT_DIR)

    # for use chroma locally, once we got docker set switch Client() -> HttpClient()
    chroma_client = chromadb.Client()

    # get_or_create_collection to avoid creating a new collection every time
    collection = chroma_client.get_or_create_collection(
        name="mini_docs",
        configuration={
            "hnsw": {  # if set chroma on cloud, probably need to change 'hnsw' -> 'spann'
                "space": "l2"
            }
        },
        metadata={  # collection metadata
            "description": "Database with docs scrapped from mini website",
            "created": str(datetime.now()),
        },
    )

    # add docs to collection
    collection.add(
        documents=docs, ids=[f"id_{i+1}" for i in range(len(docs))]
    )  # in future need to add embeddings and metadatas! e.g., hyperlinks

    return 0


if __name__ == "__main__":
    main()
