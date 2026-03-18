"""
Pydantic models and data structures for DocuMind.

Defines request/response schemas, document chunks, and session state
used across the entire RAG pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Document models
# ──────────────────────────────────────────────────────────────────────


class DocumentChunk(BaseModel):
    """A single chunk of text extracted from a PDF document."""

    id: str = Field(..., description="Unique chunk ID: {doc_hash}_{chunk_index}")
    text: str = Field(..., description="The chunk text content")
    page_number: int = Field(..., description="1-based page number in the source PDF")
    source_filename: str = Field(..., description="Original PDF filename")
    chunk_index: int = Field(..., description="0-based position in the chunk sequence")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Flatten to ChromaDB-compatible metadata (primitives only).

        Returns:
            Dict with only str/int/float/bool values.
        """
        return {
            "page_number": self.page_number,
            "source_filename": self.source_filename,
            "chunk_index": self.chunk_index,
        }


# ──────────────────────────────────────────────────────────────────────
# API request / response models
# ──────────────────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    """Response returned after uploading a PDF document."""

    success: bool
    filename: str
    num_pages: int
    num_chunks: int
    message: str


class QuestionRequest(BaseModel):
    """Incoming question from the user."""

    question: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceReference(BaseModel):
    """A source citation included in an answer."""

    source_filename: str
    page_number: int
    text_snippet: str


class AnswerResponse(BaseModel):
    """Response returned after asking a question."""

    answer: str
    sources: List[SourceReference]
    session_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    follow_up_suggestions: List[str] = Field(default_factory=list)


class SessionInfo(BaseModel):
    """Public session metadata returned via the API."""

    session_id: str
    uploaded_files: List[str]
    message_count: int
    created_at: str


# ──────────────────────────────────────────────────────────────────────
# Chat models
# ──────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    sources: Optional[List[SourceReference]] = None
    timestamp: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────
# Internal session state (mutable, not a Pydantic model)
# ──────────────────────────────────────────────────────────────────────


class SessionState:
    """Mutable in-memory session state.

    Attributes:
        session_id: Unique session identifier.
        chat_history: Ordered list of chat messages.
        uploaded_files: Filenames of uploaded PDFs.
        created_at: Unix timestamp when the session was created.
        last_accessed: Unix timestamp of the last interaction.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.chat_history: List[ChatMessage] = []
        self.uploaded_files: List[str] = []
        self.created_at: float = time.time()
        self.last_accessed: float = time.time()

    def touch(self) -> None:
        """Update the last-accessed timestamp."""
        self.last_accessed = time.time()

    def to_session_info(self) -> SessionInfo:
        """Convert to the public API model.

        Returns:
            SessionInfo with current state.
        """
        from datetime import datetime, timezone

        created_dt = datetime.fromtimestamp(
            self.created_at, tz=timezone.utc
        ).isoformat()
        return SessionInfo(
            session_id=self.session_id,
            uploaded_files=list(self.uploaded_files),
            message_count=len(self.chat_history),
            created_at=created_dt,
        )
