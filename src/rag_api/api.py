import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_api.main import query_llm
from rag_api.modules.prompt_builder import build_prompt
from rag_api.modules.retrieval import get_top_k_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.post("/chat")
def chat_endpoint(request: QueryRequest):
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
