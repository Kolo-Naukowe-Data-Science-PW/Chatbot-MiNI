"""
Shared data models for the RAG API.
"""

from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """
    Pydantic model representing a single message in conversation history.

    Attributes
    ----------
    role : Literal["user", "assistant"]
        The role of the message sender (must be "user" or "assistant").
    content : str
        The content of the message.
    """

    role: Literal["user", "assistant"]
    content: str
