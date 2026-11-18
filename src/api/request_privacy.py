"""Anonymization of API requests: glue between api.py and anonymizer.py.

Kept out of api.py on purpose: importing api.py loads the retrieval models,
this module can be unit-tested without them. See src/api/ANONYMIZATION.md.

Policy (see ANONYMIZATION.md):
- the question typed by the user: regex + local model (names of private people),
- attachments (.txt/.md/.csv pasted into the question, text extracted from PDFs):
  regex only (identifiers), file names are dropped,
- PDFs are never sent to OpenRouter as base64, only their (masked) text;
  a PDF without a text layer (scan) is rejected with HTTP 422.
If a frontend sends no attachments, the attachment code simply does nothing.
"""

from __future__ import annotations

import base64
import io
import re
from collections import Counter
from collections.abc import Callable, Iterable

from fastapi import HTTPException
from pypdf import PdfReader

from src.api.anonymizer import (
    AnonymizationResult,
    AnonymizationUnavailable,
    Anonymizer,
    TextTooLong,
    mask_identifiers,
)
from src.api.models import Message

UNAVAILABLE_MSG = {
    "pl": "Filtr prywatności jest chwilowo niedostępny. Spróbuj ponownie za chwilę.",
    "en": "The privacy filter is temporarily unavailable. Please try again in a moment.",
    "ua": "Фільтр конфіденційності тимчасово недоступний. Спробуйте ще раз за хвилину.",
}
TOO_LONG_MSG = {
    "pl": "Wiadomość jest za długa. Skróć ją albo dołącz tekst jako plik.",
    "en": "The message is too long. Please shorten it or attach the text as a file.",
    "ua": "Повідомлення задовге. Скоротіть його або додайте текст як файл.",
}
UNREADABLE_PDF_MSG = {
    "pl": "Nie udało się odczytać tekstu z pliku PDF (np. skan). Skany nie są obsługiwane: wklej treść jako tekst.",
    "en": "Could not read text from the PDF (e.g. a scan). Scans are not supported: please paste the content as text.",
    "ua": "Не вдалося прочитати текст із PDF (наприклад, скан). Скани не підтримуються: вставте вміст як текст.",
}

ATTACHMENT_LABEL = "[Załącznik]"
# How the frontend (App.jsx, handleSubmit) marks attachments in the question, with the
# word in the interface language (i18n.js, key "attachment"):
#   .txt/.md/.csv: "<question>\n\n[Załącznik: <file name>]\n<file text>"
#   PDF without a typed question: "[Załącznik: <file name>]" (the PDF itself is in `attachments`)
_ATTACHMENT_MARK = re.compile(
    r"(?:^|\n\n)\[(?:Załącznik|Attachment|Вкладення): [^\]\n]*\](?:\n|$)"
)


def split_attachment(text: str) -> tuple[str, str | None]:
    """Split a user message into (question, attachment text). No marker -> (text, None)."""
    m = _ATTACHMENT_MARK.search(text or "")
    if not m:
        return text or "", None
    return text[: m.start()], text[m.end() :]


def _mask_message(
    text: str, mask_question: Callable[[str], AnonymizationResult]
) -> tuple[str, Counter]:
    """Mask the question with ``mask_question`` and the attachment with regex; drop the file name."""
    question, attached = split_attachment(text)
    q = mask_question(question)
    found = Counter(q.found)
    if attached is None:
        return q.text, found
    parts = [q.text.rstrip()] if q.text.strip() else []
    if attached.strip():
        a = mask_identifiers(attached)
        found.update(a.found)
        parts.append(f"{ATTACHMENT_LABEL}\n{a.text}")
    else:
        parts.append(ATTACHMENT_LABEL)
    return "\n\n".join(parts), found


def anonymize_message(anonymizer: Anonymizer, text: str) -> tuple[str, Counter]:
    """A user message: question -> regex + model, attachment -> regex.

    Raises AnonymizationUnavailable (model down / text too long, fallback=block).
    """
    return _mask_message(text, anonymizer.anonymize)


class UnreadablePdf(ValueError):
    """A PDF without a text layer (scan) or a broken one: it cannot be masked, so it is not sent."""


def pdf_attachments_as_text(attachments: Iterable) -> tuple[str, Counter]:
    """Base64 PDFs -> masked text. File names are not used (they can contain a name).

    Raises UnreadablePdf for a scan without a text layer or a broken PDF: it could
    only be sent as an image, unmasked, so it is rejected instead.
    """
    parts, found = [], Counter()
    for att in attachments:
        if "pdf" not in (getattr(att, "mime_type", "") or "").lower():
            continue  # the frontend only sends PDFs; anything else is not forwarded
        try:
            reader = PdfReader(io.BytesIO(base64.b64decode(att.data)))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # noqa: BLE001  broken / encrypted PDF
            raise UnreadablePdf("PDF could not be parsed") from exc
        if not text.strip():
            raise UnreadablePdf("PDF has no text layer")
        masked = mask_identifiers(text)
        found.update(masked.found)
        parts.append(f"{ATTACHMENT_LABEL}\n{masked.text}")
    return "\n\n".join(parts), found


def _http_error(exc: AnonymizationUnavailable, lang: str | None) -> HTTPException:
    if isinstance(exc.__cause__, TextTooLong):
        return HTTPException(
            status_code=413, detail=TOO_LONG_MSG.get(lang or "pl", TOO_LONG_MSG["pl"])
        )
    return HTTPException(
        status_code=503, detail=UNAVAILABLE_MSG.get(lang or "pl", UNAVAILABLE_MSG["pl"])
    )


def anonymize_text_or_http_error(
    anonymizer: Anonymizer, text: str, lang: str | None = "pl"
) -> str:
    """Plain text (e.g. /retrieval): regex + model, HTTP 503/413 if the prompt must not be sent."""
    try:
        return anonymizer.anonymize(text).text
    except AnonymizationUnavailable as exc:
        raise _http_error(exc, lang) from exc


def anonymize_request(
    anonymizer: Anonymizer, request
) -> tuple[str, list[Message], dict[str, int]]:
    """Anonymize a /chat request BEFORE logging, translation, query rewriting and OpenRouter.

    Returns (query, conversation_history, found), where ``found`` holds only counts
    per label (safe to log). Raises HTTPException 503 (model unavailable),
    413 (question too long for the model) or 422 (PDF without text, e.g. a scan).
    """
    lang = request.language or "pl"
    # PDFs first: a rejected scan should not cost a model call
    try:
        pdf_text, pdf_found = pdf_attachments_as_text(request.attachments)
    except UnreadablePdf as exc:
        raise HTTPException(
            status_code=422,
            detail=UNREADABLE_PDF_MSG.get(lang, UNREADABLE_PDF_MSG["pl"]),
        ) from exc

    try:
        query, found = anonymize_message(anonymizer, request.query)
        history = []
        for msg in request.conversation_history:
            if msg.role == "user":
                # the frontend re-sends earlier questions verbatim: model results come from the cache
                content, _ = anonymize_message(anonymizer, msg.content)
                msg = msg.model_copy(update={"content": content})
            history.append(msg)
    except AnonymizationUnavailable as exc:
        raise _http_error(exc, request.language) from exc

    found.update(pdf_found)
    if pdf_text:
        query = (
            pdf_text
            if query.strip() in ("", ATTACHMENT_LABEL)
            else f"{query}\n\n{pdf_text}"
        )
    return query, history, dict(found)
