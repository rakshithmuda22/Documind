"""
RAG Pipeline orchestrator for DocuMind.

Ties together PDF processing, embeddings, vector storage, and LLM
generation into a cohesive document Q&A pipeline with session management.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from embeddings import EmbeddingService
from llm_service import LLMService
from models import (
    AnswerResponse,
    ChatMessage,
    SessionState,
    SourceReference,
    UploadResponse,
)
from pdf_processor import compute_doc_hash, process_pdf
from vector_store import VectorStoreService

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MAX_CHAT_HISTORY: int = 20
SESSION_TIMEOUT_SECONDS: int = 2 * 60 * 60  # 2 hours
SNIPPET_LENGTH: int = 150


class RAGPipeline:
    """Orchestrates the full RAG document Q&A workflow.

    Manages user sessions, document ingestion, and the
    retrieve-then-generate pipeline for answering questions.

    Attributes:
        embedding_service: Shared embedding model instance.
        vector_store: ChromaDB vector store.
        llm_service: Groq LLM client.
        sessions: In-memory session state mapping.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        """Initialise the RAG pipeline with optional dependency injection.

        Args:
            embedding_service: Embedding service (created if None).
            vector_store: Vector store (created if None).
            llm_service: LLM service (created if None).
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()
        self.llm_service = llm_service or LLMService()
        self.sessions: Dict[str, SessionState] = {}
        logger.info("RAGPipeline initialised")

    # ──────────────────────────────────────────────────────────────────
    # Session management
    # ──────────────────────────────────────────────────────────────────

    def create_session(self) -> str:
        """Create a new session.

        Returns:
            A unique session ID (UUID4 hex string).
        """
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = SessionState(session_id)
        logger.info("Created session %s", session_id[:8])
        return session_id

    def get_session(self, session_id: str) -> SessionState:
        """Retrieve an existing session.

        Args:
            session_id: The session identifier.

        Returns:
            The SessionState object.

        Raises:
            ValueError: If the session does not exist.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(
                f"Session {session_id[:8]}... not found. "
                "It may have expired. Please start a new session."
            )
        session.touch()
        return session

    def clear_session(self, session_id: str) -> None:
        """Delete a session and its associated vector data.

        Args:
            session_id: The session identifier.
        """
        self.vector_store.delete_collection(session_id)
        self.sessions.pop(session_id, None)
        logger.info("Cleared session %s", session_id[:8])

    def cleanup_expired_sessions(self) -> int:
        """Remove sessions that have been inactive for too long.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        expired = [
            sid for sid, state in self.sessions.items()
            if (now - state.last_accessed) > SESSION_TIMEOUT_SECONDS
        ]
        for sid in expired:
            self.clear_session(sid)

        if expired:
            logger.info("Cleaned up %d expired session(s)", len(expired))
        return len(expired)

    # ──────────────────────────────────────────────────────────────────
    # Document ingestion
    # ──────────────────────────────────────────────────────────────────

    async def ingest_document(
        self,
        session_id: str,
        file_bytes: bytes,
        filename: str,
    ) -> UploadResponse:
        """Process and index a PDF document for a session.

        Steps: validate → extract text → chunk → embed → store in vector DB.

        Args:
            session_id: The session identifier.
            file_bytes: Raw PDF file content.
            filename: Original filename.

        Returns:
            UploadResponse with processing results.

        Raises:
            ValueError: If the session doesn't exist or PDF is invalid.
        """
        session = self.get_session(session_id)

        # Process PDF into chunks
        chunks = process_pdf(file_bytes, filename)

        if not chunks:
            raise ValueError(
                f"No text could be extracted from '{filename}'. "
                "The PDF may be scanned or image-based."
            )

        # Check embedding cache
        doc_hash = compute_doc_hash(file_bytes)
        cached = self.embedding_service.get_cached_embeddings(doc_hash)

        if cached and len(cached) == len(chunks):
            embeddings = cached
            logger.info("Using cached embeddings for %s", filename)
        else:
            # Embed all chunks
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            self.embedding_service.cache_embeddings(doc_hash, embeddings)

        # Store in vector database
        self.vector_store.add_documents(session_id, chunks, embeddings)

        # Update session state
        if filename not in session.uploaded_files:
            session.uploaded_files.append(filename)

        # Count pages
        page_numbers = {chunk.page_number for chunk in chunks}

        logger.info(
            "Ingested '%s': %d pages, %d chunks for session %s",
            filename, len(page_numbers), len(chunks), session_id[:8],
        )

        return UploadResponse(
            success=True,
            filename=filename,
            num_pages=len(page_numbers),
            num_chunks=len(chunks),
            message=f"Successfully processed '{filename}': "
                    f"{len(page_numbers)} pages, {len(chunks)} chunks indexed.",
        )

    # ──────────────────────────────────────────────────────────────────
    # Question answering
    # ──────────────────────────────────────────────────────────────────

    async def answer_question(
        self,
        session_id: str,
        question: str,
        top_k: int = 5,
    ) -> AnswerResponse:
        """Answer a question using the RAG pipeline.

        Steps: detect follow-up → rewrite → embed → retrieve → generate.

        Args:
            session_id: The session identifier.
            question: The user's question.
            top_k: Number of context chunks to retrieve.

        Returns:
            AnswerResponse with answer, sources, and confidence.

        Raises:
            ValueError: If the session doesn't exist or has no documents.
        """
        session = self.get_session(session_id)

        # Verify documents have been uploaded
        if not self.vector_store.collection_exists(session_id):
            raise ValueError(
                "No documents have been uploaded yet. "
                "Please upload a PDF before asking questions."
            )

        # Build chat history for LLM
        history = self._get_chat_history_dicts(session)

        # Step 1: Detect follow-up and rewrite if needed
        search_question = question
        if history:
            followup_result = await self.llm_service.detect_follow_up(
                question, history,
            )
            if followup_result["is_followup"]:
                search_question = followup_result["standalone_question"]
                logger.info(
                    "Rewrote follow-up: '%s' → '%s'",
                    question[:50], search_question[:50],
                )

        # Step 2: Embed the query
        query_embedding = self.embedding_service.embed_query(search_question)

        # Step 3: Retrieve relevant chunks
        retrieved = self.vector_store.search(
            session_id, query_embedding, top_k=top_k,
        )

        if not retrieved:
            return AnswerResponse(
                answer="I couldn't find any relevant information in the "
                       "uploaded documents to answer your question.",
                sources=[],
                session_id=session_id,
                confidence=0.0,
            )

        # Step 4: Generate answer with LLM
        llm_result = await self.llm_service.answer_question(
            question, retrieved, history,
        )

        # Step 5: Build source references
        sources = self._build_sources(retrieved, llm_result.get("sources_used", []))

        # Step 6: Update chat history
        session.chat_history.append(ChatMessage(
            role="user",
            content=question,
            timestamp=time.time(),
        ))
        session.chat_history.append(ChatMessage(
            role="assistant",
            content=llm_result["answer"],
            sources=sources,
            timestamp=time.time(),
        ))

        # Trim chat history to prevent unbounded growth
        if len(session.chat_history) > MAX_CHAT_HISTORY:
            session.chat_history = session.chat_history[-MAX_CHAT_HISTORY:]

        return AnswerResponse(
            answer=llm_result["answer"],
            sources=sources,
            session_id=session_id,
            confidence=llm_result["confidence"],
            follow_up_suggestions=llm_result.get("follow_up_suggestions", []),
        )

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_chat_history_dicts(
        session: SessionState,
    ) -> List[Dict[str, str]]:
        """Convert session chat history to simple dicts for prompts.

        Args:
            session: The current session state.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in session.chat_history
        ]

    @staticmethod
    def _build_sources(
        retrieved: List[Dict[str, Any]],
        sources_used: List[int],
    ) -> List[SourceReference]:
        """Build source references from retrieved chunks.

        Args:
            retrieved: All retrieved chunks from vector search.
            sources_used: Indices of chunks the LLM actually used.

        Returns:
            List of SourceReference objects.
        """
        sources = []

        # If LLM specified which sources it used, prioritise those
        if sources_used:
            indices = [
                i for i in sources_used
                if isinstance(i, int) and 0 <= i < len(retrieved)
            ]
        else:
            # Fall back to all retrieved chunks
            indices = list(range(len(retrieved)))

        for i in indices:
            chunk = retrieved[i]
            snippet = chunk["text"][:SNIPPET_LENGTH]
            if len(chunk["text"]) > SNIPPET_LENGTH:
                snippet += "..."

            sources.append(SourceReference(
                source_filename=chunk["source_filename"],
                page_number=chunk["page_number"],
                text_snippet=snippet,
            ))

        return sources
