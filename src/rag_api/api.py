import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_api.main import query_llm
from rag_api.modules.prompt_builder import build_prompt
from rag_api.modules.retrieval import get_top_k_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class QueryRequest(BaseModel):
    """
    Pydantic model representing the incoming chat request.

    Attributes
    ----------
    query : str
        The question or text input provided by the user.
    """

    query: str


@app.post("/chat")
def chat_endpoint(request: QueryRequest) -> dict[str, Any]:
    """
    Handles chat interactions by retrieving context and generating an LLM response.

    1. Validates the input query.
    2. Retrieves the top-k relevant text chunks from the vector database.
    3. Builds a prompt using the retrieved context.
    4. Queries the LLM to generate an answer.
    5. Returns the answer along with source URLs.

    Parameters
    ----------
    request : QueryRequest
        The request body containing the user's query.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - 'answer': The generated response string.
        - 'sources': A list of source URLs used for the context.

    Raises
    ------
    HTTPException
        If the query is empty (400 Bad Request).
    """
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Received query: {query}")

    sorted_chunks = get_top_k_chunks(query)

    if not sorted_chunks:
        return {
            "answer": "Przepraszam, nie znalazłem w bazie informacji na ten temat.",
            "sources": [],
        }

    text_only_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]
    prompt = build_prompt(query, text_only_chunks)

    answer = query_llm(prompt)

    sources = [chunk.get("source_url", "Unknown") for chunk in sorted_chunks]

    return {"answer": answer, "sources": sources}
