"""
FastAPI module endpoint (/chat).
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_api.main import query_llm
from src.rag_api.modules.logs import setup_logging
from src.rag_api.modules.prompt_builder import build_prompt
from src.rag_api.modules.retrieval import get_top_k_chunks

setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI(title="ChatBot-Mini API", version="1.0.0")


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Handle user question through the RAG pipeline.
    1. Recive a question from the user.
    2. Process the question to retrieve relevant document chunks.
    3. Build a prompt.
    4. Query the LLM.
    5. Return answer and sources.
    """
    logger.info("Received chat request: '%s'", request.query)

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Zapytanie nie może być puste.")

    try:
        retrieval_result = get_top_k_chunks(request.query, top_k=request.top_k)

        docs = retrieval_result.get("documents", [[]])[0]
        metas = retrieval_result.get("metadatas", [[]])[0]

        if not isinstance(docs, list) or not isinstance(metas, list):
            raise ValueError("Invalid retrieval result format")

        if len(docs) != len(metas):
            raise HTTPException(
                status_code=500,
                detail="Invalid retrieval format: docs and metas lengths differ.",
            )

        text_chunks = []
        sources = []
        for doc, meta in zip(docs, metas, strict=False):
            source_url = meta.get("source_url", "Unknown source")
            text_chunks.append(doc)
            sources.append(
                {"source_url": source_url, "text_excerpt": doc[:200] + "..."}
            )

        prompt = build_prompt(request.query, text_chunks)
        answer = query_llm(prompt)

        logger.info("Answer successfully generated for query.")

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error("Error in /chat endpoint: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
