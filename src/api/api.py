import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.main import query_llm
from src.api.models import Message
from src.api.prompt_builder import build_messages
from src.api.query_rewriter import rewrite_query
from src.api.retrieval import get_top_k_chunks
from src.api.translator import translate_text
from src.utils.paths import get_data_dir

# minor change just to trigger deployment again
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
feedback_write_lock = Lock()
feedback_file_path = Path(get_data_dir("feedback", "model_feedback.csv"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """
    Pydantic model representing the structure of a chat query request.
    Attributes: - query: The user's input query string.
                - language: The language code of the query (default is "pl" for Polish).
                - conversation_history: A list of Message objects representing the prior conversation context.
                - mode: Optional string indicating the frontend mode (e.g., "production", "testPro").
                - variant: Optional string indicating the A/B test variant (e.g., "A", "B").
                - modelConfig: Optional dictionary containing model parameters used for generating the response.
    """

    query: str
    language: str = "pl"
    conversation_history: list[Message] = []
    mode: str | None = None
    variant: str | None = None
    modelConfig: dict | None = None
    user_type: str | None = None


class FeedbackRequest(BaseModel):
    """
    Pydantic model representing a feedback event from the frontend.
    Attributes: - message_id: Optional integer ID of the message pair (query-response) being rated.
                - pair_id: Optional integer ID representing the conversation pair (could be same as message_id or a separate identifier).
                - variant_label: Optional string indicating the A/B test variant (e.g., "A", "B").
                - version: Optional string indicating the version of the model or system.
                - language: Optional string indicating the language of the query/response.
                - rating: Optional integer or string representing the user's rating
                - ratings: Dict with integers representing the user's rating
                - selected: Optional boolean indicating if this response was selected as the best answer among alternatives.
                - query: Optional string of the original user query.
                - response_text: Optional string of the LLM-generated response text.
                - variant_config: Optional dictionary containing the model parameters used for generating the response.
                - selection_timestamp: Optional string timestamp of when the response was selected.
                - rating_timestamp: Optional string timestamp of when the rating was given.
                - created_at: Optional string timestamp of when the feedback event was created
    """

    message_id: int | None = None
    pair_id: int | None = None
    variant_label: str | None = None
    version: str | None = None
    language: str | None = None
    user_type: str | None = None
    rating: int | str | None = None
    ratings: dict[str, int] | None = None
    selected: bool | None = None
    query: str | None = None
    response_text: str | None = None
    variant_config: dict[str, Any] | None = None
    selection_timestamp: str | None = None
    rating_timestamp: str | None = None
    created_at: str | None = None


def create_row(payload: FeedbackRequest, variant_config):
    """
    The function `create_row_without_ratings` creates a dictionary row from a `FeedbackRequest` payload
    without including the rating information.

    """

    row = {
        "created_at": payload.created_at or datetime.now(UTC).isoformat(),
        "message_id": payload.message_id or "",
        "pair_id": payload.pair_id or "",
        "variant_label": payload.variant_label or "",
        "version": payload.version or "",
        "language": payload.language or "",
        "user_type": payload.user_type or "",
        "rating": payload.rating if payload.rating is not None else "",
        "ratings": payload.ratings or None,
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
    return row


def append_feedback_row(payload: FeedbackRequest) -> None:
    """
    Append one feedback record to the CSV storage file.
    Input: FeedbackRequest payload containing all relevant feedback information.
    Output: None (side effect is writing a new row to the CSV file).
    """

    feedback_file_path.parent.mkdir(parents=True, exist_ok=True)
    variant_config = payload.variant_config or {}
    row = create_row(payload, variant_config)

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

    retrieval_query = rewrite_query(processing_query)
    sorted_chunks = get_top_k_chunks(retrieval_query)

    if not sorted_chunks:
        polish_msg = "Przepraszam, nie znalazłem w bazie informacji na ten temat."
        final_msg = translate_text(polish_msg, lang) if lang != "pl" else polish_msg

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
        processing_query,
        text_only_chunks,
        user_type=request.user_type,
        conversation_history=conversation_history,
    )

    polish_answer = query_llm(messages, request.modelConfig)
    final_answer = polish_answer
    if lang != "pl":
        logger.info(f"Translating answer from PL to {lang}...")
        final_answer = translate_text(polish_answer, target_lang_code=lang)

    sources = [chunk.get("source_url", "Unknown") for chunk in sorted_chunks[:5]]

    return {"answer": final_answer, "sources": sources, "retrieval_query": retrieval_query}


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
