# dodane: zapis feedbacku modeli do CSV
import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException

# dodane: do pozwoleń CORS
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_api.main import query_llm
from src.rag_api.models import Message
from src.rag_api.modules.prompt_builder import build_messages
from src.rag_api.modules.retrieval import get_top_k_chunks
from src.rag_api.modules.translator import translate_text
from src.utils.paths import get_data_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
# dodane: lock i ścieżka pliku CSV z feedbackiem
feedback_write_lock = Lock()
feedback_file_path = Path(get_data_dir("feedback", "model_feedback.csv"))

# dodane: pozwolenia CORS (UWAGA: czy ograniczamy?)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# dodane: model requestu do feedback
class FeedbackRequest(BaseModel):
    """
    Pydantic model representing a feedback event from the frontend.

    Attributes
    ----------
    message_id : int | None
        Unique identifier of the rated message.
    pair_id : int | None
        Identifier binding variant A/B responses to the same prompt.
    variant_label : str | None
        Variant name (for example ``A`` or ``B``).
    version : str | None
        Active frontend mode (for example ``production`` or ``testPro``).
    language : str | None
        UI language code used during the interaction.
    rating : int | str | None
        User rating value (thumbs or stars depending on mode).
    selected : bool | None
        Whether the user selected this variant as preferred.
    query : str | None
        User prompt text.
    response_text : str | None
        Assistant response text being rated.
    variant_config : dict[str, Any] | None
        Model and generation parameters associated with the variant.
    selection_timestamp : str | None
        ISO timestamp when a variant was selected.
    rating_timestamp : str | None
        ISO timestamp when a rating was submitted.
    created_at : str | None
        Event creation timestamp from the frontend.
    """

    message_id: int | None = None
    pair_id: int | None = None
    variant_label: str | None = None
    version: str | None = None
    language: str | None = None
    rating: int | str | None = None
    selected: bool | None = None
    query: str | None = None
    response_text: str | None = None
    variant_config: dict[str, Any] | None = None
    selection_timestamp: str | None = None
    rating_timestamp: str | None = None
    created_at: str | None = None


# dodane: funkcja dopisująca rekord feedbacku do CSV
def append_feedback_row(payload: FeedbackRequest) -> None:
    """
    Append one feedback record to the CSV storage file.

    Parameters
    ----------
    payload : FeedbackRequest
        Feedback event sent by the frontend.

    Returns
    -------
    None
        This function persists data and does not return a value.
    """

    feedback_file_path.parent.mkdir(parents=True, exist_ok=True)
    variant_config = payload.variant_config or {}
    row = {
        "created_at": payload.created_at or datetime.now(UTC).isoformat(),
        "message_id": payload.message_id or "",
        "pair_id": payload.pair_id or "",
        "variant_label": payload.variant_label or "",
        "version": payload.version or "",
        "language": payload.language or "",
        "rating": payload.rating if payload.rating is not None else "",
        "selected": payload.selected if payload.selected is not None else "",
        "selection_timestamp": payload.selection_timestamp or "",
        "rating_timestamp": payload.rating_timestamp or "",
        "model": variant_config.get("model", ""),
        "temperature": variant_config.get("temperature", ""),
        "top_p": variant_config.get("top_p", ""),
        "frequency_penalty": variant_config.get("frequency_penalty", ""),
        "presence_penalty": variant_config.get("presence_penalty", ""),
        "max_tokens": variant_config.get("max_tokens", ""),
        "query": payload.query or "",
        "response_text": payload.response_text or "",
    }
    fieldnames = list(row.keys())
    file_exists = feedback_file_path.exists()
    with feedback_write_lock:
        with feedback_file_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


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
        # Store Polish versions to match what LLM processes
        updated_history = conversation_history + [
            Message(role="user", content=processing_query),
            Message(role="assistant", content=polish_msg),
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

    return {"answer": final_answer, "sources": sources}


# dodane: endpoint zapisujący feedback do CSV
@app.post("/feedback")
def feedback_endpoint(payload: FeedbackRequest) -> dict[str, str]:
    """
    Persist a single feedback event and return status information.

    Parameters
    ----------
    payload : FeedbackRequest
        Feedback payload received from the frontend client.

    Returns
    -------
    dict[str, str]
        A dictionary with operation status.
    """

    append_feedback_row(payload)
    return {"status": "ok"}
    # Update conversation history with new exchange
    # Store Polish versions to match what LLM processes
    updated_history = conversation_history + [
        Message(role="user", content=processing_query),
        Message(role="assistant", content=polish_answer),
    ]

    return {
        "answer": final_answer,
        "sources": sources,
        "conversation_history": updated_history,
    }
