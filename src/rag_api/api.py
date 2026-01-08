import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_api.main import query_llm
from src.rag_api.models import Message
from src.rag_api.modules.prompt_builder import build_messages
from src.rag_api.modules.retrieval import get_top_k_chunks
from src.rag_api.modules.translator import translate_text

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
    language : str
        The language code for the response (default: "pl").
    conversation_history : list[Message]
        The conversation history from previous interactions in this session.
    """

    query: str
    language: str = "pl"
    conversation_history: list[Message] = []


@app.post("/chat")
def chat_endpoint(request: QueryRequest) -> dict[str, Any]:
    """
    Handles chat interactions by retrieving context and generating an LLM response.

    1. Validates the input query.
    2. Retrieves the top-k relevant text chunks from the vector database.
    3. Builds a prompt using the retrieved context and conversation history.
    4. Queries the LLM to generate an answer.
    5. Returns the answer along with source URLs and updated conversation history.

    Parameters
    ----------
    request : QueryRequest
        The request body containing the user's query and conversation history.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - 'answer': The generated response string.
        - 'sources': A list of source URLs used for the context.
        - 'conversation_history': Updated conversation history including the new exchange.

    Raises
    ------
    HTTPException
        If the query is empty (400 Bad Request).
    """
    query = request.query
    lang = request.language
    conversation_history = request.conversation_history

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(
        f"Received query: {query} | Target lang: {lang} | History length: {len(conversation_history)}"
    )

    processing_query = query
    if lang != "pl":
        processing_query = translate_text(query, target_lang_code="pl")
        logger.info(f"Translated query to PL: '{processing_query}'")

    sorted_chunks = get_top_k_chunks(processing_query)

    if not sorted_chunks:
        polish_msg = "Przepraszam, nie znalazłem w bazie informacji na ten temat."
        final_msg = translate_text(polish_msg, lang) if lang != "pl" else polish_msg

        # Update conversation history even when no results found
        updated_history = conversation_history + [
            Message(role="user", content=query),
            Message(role="assistant", content=final_msg),
        ]

        return {
            "answer": final_msg,
            "sources": [],
            "conversation_history": updated_history,
        }

    text_only_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]

    messages = build_messages(
        processing_query, text_only_chunks, conversation_history=conversation_history
    )

    polish_answer = query_llm(messages)
    final_answer = polish_answer
    if lang != "pl":
        logger.info(f"Translating answer from PL to {lang}...")
        final_answer = translate_text(polish_answer, target_lang_code=lang)

    sources = [chunk.get("source_url", "Unknown") for chunk in sorted_chunks[:5]]

    # Update conversation history with new exchange
    updated_history = conversation_history + [
        Message(role="user", content=query),
        Message(role="assistant", content=final_answer),
    ]

    return {
        "answer": final_answer,
        "sources": sources,
        "conversation_history": updated_history,
    }
