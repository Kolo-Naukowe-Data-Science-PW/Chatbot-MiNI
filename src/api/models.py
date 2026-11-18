"""
Shared data models for the RAG API.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    """
    A file attached to a chat message.

    Preferred format (returned by ``POST /attachments``): already processed
    content — ``text`` for documents and/or ``images`` (JPEG data URLs).

    Legacy format: raw ``data`` (base64) + ``mime_type``; the API processes it
    on the fly with the same code as ``/attachments``.
    """

    filename: str = "plik"
    kind: str | None = None
    text: str | None = None
    images: list[str] = Field(default_factory=list)
    note: str | None = None
    truncated: bool = False
    pages: int | None = None
    # legacy fields
    data: str | None = None
    mime_type: str | None = None


class Message(BaseModel):
    """
    Pydantic model representing a single message in conversation history.

    Attributes
    ----------
    role : Literal["user", "assistant"]
        The role of the message sender (must be "user" or "assistant").
    content : str
        The content of the message.
    attachments : list[Attachment]
        Files attached to this (user) message. Only their text is replayed
        to the model in later turns; images are not re-sent.
    """

    role: Literal["user", "assistant"]
    content: str
    attachments: list[Attachment] = Field(default_factory=list)
