import csv
import io
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.main import query_llm, query_llm_stream
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


class FileAttachment(BaseModel):
    filename: str
    data: str  # base64-encoded file content
    mime_type: str = "application/pdf"


class QueryRequest(BaseModel):
    """
    Pydantic model representing the structure of a chat query request.
    Attributes: - query: The user's input query string.
                - language: The language code of the query (default is "pl" for Polish).
                - conversation_history: A list of Message objects representing the prior conversation context.
                - mode: Optional string indicating the frontend mode (e.g., "production", "testPro").
                - variant: Optional string indicating the A/B test variant (e.g., "A", "B").
                - modelConfig: Optional dictionary containing model parameters used for generating the response.
                - attachments: Optional list of file attachments (e.g., PDFs) sent as base64 to the model.
    """

    query: str
    language: str = "pl"
    conversation_history: list[Message] = []
    mode: str | None = None
    variant: str | None = None
    modelConfig: dict | None = None
    user_type: str | None = None
    major: str | None = None
    semester: str | None = None
    attachments: list[FileAttachment] = []


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


_ROMAN_TO_ARABIC = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", "VII": "7"}

# Keywords that indicate a schedule-related query; only these get major/semester enrichment.
# Enriching every query causes schedule documents to rank high for unrelated questions.
_SCHEDULE_KEYWORDS = (
    "plan zajęć", "harmonogram", "siatka zajęć", "wykład", "ćwiczeni",
    "laboratorium", "godziny zajęć", "sala ", "kiedy są zajęcia",
    "kiedy mam zajęcia", "kiedy mam wykład",
    "jakie zajęcia mam", "jakie mam zajęcia", "zajęcia",
    "jaki mam plan", "jaki plan jest",
    "w jakiej sali mam", "w jakiej sali",
    "kto prowadzi", "prowadzący",
)

_CURRICULUM_KEYWORDS = (
    "plan studiów", "plan studiow", "program studiów", "program studiow",
    "siatka studiów", "ile ects", "punkty ects", "ile punktów",
    "przedmioty obowiązkowe", "przedmioty w semestrze", "przedmioty na semestrze",
)

_NEXT_SEM_KEYWORDS = (
    "następny semestr", "następnym semestrze", "następnego semestru",
    "kolejny semestr", "kolejnym semestrze", "kolejnego semestru",
    "przyszły semestr", "przyszłym semestrze",
)


def _is_schedule_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _SCHEDULE_KEYWORDS)


def _enrich_retrieval_query(
    query: str, major: str | None, semester: str | None, original_query: str | None = None
) -> str:
    """Append major/semester context to schedule and curriculum queries.

    Schedule queries get current semester; curriculum/next-semester queries get
    next semester (so 'następny semestr' for a sem-4 student resolves to sem 5).
    'plan' alone is intentionally excluded from schedule keywords — it also matches
    'plan studiów' which is a curriculum query.
    """
    if not major or major == "—":
        return query
    orig = (original_query or query).lower()
    q_lower = query.lower()

    if _is_schedule_query(query):
        q = query + f" kierunek {major}"
        if semester and semester != "—":
            arabic = _ROMAN_TO_ARABIC.get(semester, semester)
            q += f" semestr {arabic}"
        return q

    is_curriculum = any(kw in q_lower for kw in _CURRICULUM_KEYWORDS)
    is_next_sem = any(kw in orig for kw in _NEXT_SEM_KEYWORDS)
    if is_curriculum or is_next_sem:
        q = query + f" kierunek {major}"
        if semester and semester != "—" and is_next_sem:
            arabic = _ROMAN_TO_ARABIC.get(semester, semester)
            try:
                q += f" semestr {int(arabic) + 1}"
            except (ValueError, TypeError):
                pass
        return q

    return query


_EXPERIMENT_PERSONAS = [
    "Odpowiedz bardzo krótko i konkretnie. Bez owijania w bawełnę.",
    "Odpowiedz luzno, prosto i przyjaźnie.",
    "Odpowiedz formalnie i akademicko, pełnymi zdaniami.",
    "Podaj wyczerpującą odpowiedź z detalami i przykładami.",
]

_EXPERIMENT_MODEL_POOL = [
    # mid-tier
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324",
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    # supermodels
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "google/gemini-3.1-pro-preview-customtools",
]


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 10
    use_rerank: bool = True
    use_rewrite: bool = False


@app.post("/retrieval")
def retrieval_endpoint(request: RetrievalRequest) -> dict[str, Any]:
    """Direct retrieval endpoint for ablation experiments (bypasses LLM generation)."""
    q = rewrite_query(request.query) if request.use_rewrite else request.query
    chunks = get_top_k_chunks(q, top_k=request.top_k, use_rerank=request.use_rerank)
    seen: set[str] = set()
    urls: list[str] = []
    for chunk in chunks:
        url = chunk.get("source_url", "")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return {"retrieval_query": q, "urls": urls}


@app.get("/db-urls")
def db_urls_endpoint() -> dict[str, Any]:
    """Return all unique source URLs stored in Qdrant. Used by coverage analysis."""
    from src.api.retrieval import _get_qdrant_client  # noqa: PLC0415
    from src.ingestion.vector_db import COLLECTION_NAME  # noqa: PLC0415

    client = _get_qdrant_client()
    urls: set[str] = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1000,
            with_payload=["url"],
            with_vectors=False,
        )
        for p in points:
            url = (p.payload or {}).get("url", "")
            if url:
                urls.add(url)
        if next_offset is None:
            break
        offset = next_offset
    return {"urls": sorted(urls), "count": len(urls)}


@app.get("/experiment-config")
def experiment_config() -> dict:
    """Return the current A/B experiment configuration from environment variables.

    Used by the frontend to know which dimension to vary (model / temperature / persona)
    and what the baseline values are for the fixed dimensions.
    """
    persona_idx = int(os.getenv("EXPERIMENT_PERSONA", "3"))
    persona_idx = max(0, min(persona_idx, len(_EXPERIMENT_PERSONAS) - 1))
    baseline_temp = float(os.getenv("EXPERIMENT_TEMP", "0.2"))
    return {
        "dim": os.getenv("EXPERIMENT_DIM", "model").lower(),
        "baseline_model": os.getenv("EXPERIMENT_MODEL", "openai/gpt-4o-mini"),
        "baseline_temp": baseline_temp,
        "persona_idx": persona_idx,
        "baseline_persona": _EXPERIMENT_PERSONAS[persona_idx],
        "personas": _EXPERIMENT_PERSONAS,
        "model_pool": _EXPERIMENT_MODEL_POOL,
    }


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

    retrieval_query = _enrich_retrieval_query(
        rewrite_query(processing_query), request.major, request.semester,
        original_query=processing_query,
    )
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

    style_instruction = (request.modelConfig or {}).get("styleInstruction")
    messages = build_messages(
        processing_query,
        text_only_chunks,
        user_type=request.user_type,
        field_of_study=request.major,
        semester=str(request.semester) if request.semester else None,
        conversation_history=conversation_history,
        style_instruction=style_instruction,
        attachments=[{"filename": a.filename, "data": a.data, "mime_type": a.mime_type} for a in request.attachments],
    )

    polish_answer = query_llm(messages, request.modelConfig)
    final_answer = polish_answer
    if lang != "pl":
        logger.info(f"Translating answer from PL to {lang}...")
        final_answer = translate_text(polish_answer, target_lang_code=lang)

    seen: set[str] = set()
    sources = []
    for chunk in sorted_chunks:
        url = chunk.get("source_url", "Unknown")
        if url not in seen:
            seen.add(url)
            sources.append(url)

    return {"answer": final_answer, "sources": sources, "retrieval_query": retrieval_query}


@app.post("/chat/stream")
def chat_stream_endpoint(request: QueryRequest):
    """
    Streaming version of /chat using Server-Sent Events (SSE).

    Yields text chunks as they arrive from the LLM, then sends a final
    JSON event with sources and the rewritten retrieval query.

    SSE event format
    ----------------
    - Text delta:  ``data: {token text}\\n\\n``
    - Final event: ``data: {"event":"done","sources":[...],"retrieval_query":"..."}\\n\\n``

    The non-streaming ``/chat`` endpoint is preserved unchanged.
    """
    query = request.query
    lang = request.language
    conversation_history = request.conversation_history

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"[stream] query='{query}' lang={lang}")

    processing_query = query
    if lang != "pl":
        processing_query = translate_text(query, target_lang_code="pl")

    retrieval_query = _enrich_retrieval_query(
        rewrite_query(processing_query), request.major, request.semester,
        original_query=processing_query,
    )
    sorted_chunks = get_top_k_chunks(retrieval_query)
    seen: set[str] = set()
    sources = []
    for chunk in sorted_chunks:
        url = chunk.get("source_url", "Unknown")
        if url not in seen:
            seen.add(url)
            sources.append(url)

    if not sorted_chunks:
        polish_msg = "Przepraszam, nie znalazłem w bazie informacji na ten temat."
        final_msg = translate_text(polish_msg, lang) if lang != "pl" else polish_msg

        def _empty():
            yield f"data: {final_msg}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'sources': [], 'retrieval_query': retrieval_query})}\n\n"

        return StreamingResponse(_empty(), media_type="text/event-stream")

    text_only_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]
    style_instruction = (request.modelConfig or {}).get("styleInstruction")
    messages = build_messages(
        processing_query,
        text_only_chunks,
        user_type=request.user_type,
        field_of_study=request.major,
        semester=str(request.semester) if request.semester else None,
        conversation_history=conversation_history,
        style_instruction=style_instruction,
        attachments=[{"filename": a.filename, "data": a.data, "mime_type": a.mime_type} for a in request.attachments],
    )

    def _sse(text: str) -> str:
        """Encode a text chunk as an SSE data line, escaping embedded newlines."""
        return f"data: {text.replace(chr(10), chr(92) + 'n').replace(chr(13), '')}\n\n"

    def _generate():
        polish_tokens: list[str] = []
        for token in query_llm_stream(messages, request.modelConfig):
            polish_tokens.append(token)
            # Stream in the original language; if translation needed we accumulate
            # and translate only the final answer (translation requires full text).
            if lang == "pl":
                yield _sse(token)

        if lang != "pl":
            polish_answer = "".join(polish_tokens)
            final_answer = translate_text(polish_answer, target_lang_code=lang)
            yield _sse(final_answer)

        yield f"data: {json.dumps({'event': 'done', 'sources': sources, 'retrieval_query': retrieval_query})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


class ErrorReportRequest(BaseModel):
    message_id: int | None = None
    response_text: str | None = None
    error_description: str
    created_at: str | None = None


@app.post("/errors")
def error_report_endpoint(payload: ErrorReportRequest) -> dict[str, str]:
    error_dir = Path(get_data_dir("errors"))
    error_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    error_file = error_dir / f"error_{timestamp}.json"
    data = {
        "created_at": payload.created_at or datetime.now(UTC).isoformat(),
        "message_id": payload.message_id,
        "response_text": payload.response_text,
        "error_description": payload.error_description,
    }
    with error_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"status": "ok"}


@app.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
    """Extract plain text from an uploaded file (.txt, .md, .pdf). Used by the frontend to attach files to chat."""
    content = await file.read()
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in ("txt", "md", "csv"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")
    elif ext == "pdf":
        try:
            from pypdf import PdfReader  # noqa: PLC0415
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"PDF extraction failed: {exc}") from exc
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: .{ext}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="File appears to be empty or unreadable.")

    return {"text": text, "filename": filename}


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
