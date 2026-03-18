"""Tests for rag_pipeline module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest

from models import AnswerResponse, UploadResponse
from rag_pipeline import RAGPipeline


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def make_test_pdf(text: str = "This is a test document with enough text to be valid.") -> bytes:
    """Create a minimal PDF with the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_mock_embedding_service():
    """Create a mock EmbeddingService."""
    mock = MagicMock()
    mock.embed_texts.return_value = [[0.1] * 384]
    mock.embed_query.return_value = [0.1] * 384
    mock.get_cached_embeddings.return_value = None
    mock.cache_embeddings.return_value = None
    return mock


def make_mock_llm_service():
    """Create a mock LLMService with async methods."""
    mock = MagicMock()
    mock.answer_question = AsyncMock(return_value={
        "answer": "The document discusses testing.",
        "confidence": 0.85,
        "sources_used": [0],
        "follow_up_suggestions": ["What else does it say?"],
    })
    mock.detect_follow_up = AsyncMock(return_value={
        "is_followup": False,
        "standalone_question": "What is this about?",
    })
    return mock


def make_mock_vector_store():
    """Create a mock VectorStoreService."""
    mock = MagicMock()
    mock.add_documents.return_value = None
    mock.collection_exists.return_value = True
    mock.search.return_value = [
        {
            "text": "This is a test document with enough text.",
            "source_filename": "test.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "distance": 0.15,
        }
    ]
    mock.delete_collection.return_value = None
    return mock


# ──────────────────────────────────────────────────────────────────────
# Tests — session management
# ──────────────────────────────────────────────────────────────────────


class TestSessionManagement:
    """Tests for session creation and lifecycle."""

    def setup_method(self):
        self.pipeline = RAGPipeline(
            embedding_service=make_mock_embedding_service(),
            vector_store=make_mock_vector_store(),
            llm_service=make_mock_llm_service(),
        )

    def test_create_session(self):
        sid = self.pipeline.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 32  # UUID4 hex

    def test_get_session(self):
        sid = self.pipeline.create_session()
        session = self.pipeline.get_session(sid)
        assert session.session_id == sid

    def test_get_nonexistent_session_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.pipeline.get_session("nonexistent")

    def test_clear_session(self):
        sid = self.pipeline.create_session()
        self.pipeline.clear_session(sid)
        with pytest.raises(ValueError):
            self.pipeline.get_session(sid)

    def test_cleanup_expired_sessions(self):
        sid = self.pipeline.create_session()
        # Manually set created time to past
        self.pipeline.sessions[sid].last_accessed = 0
        removed = self.pipeline.cleanup_expired_sessions()
        assert removed == 1
        assert sid not in self.pipeline.sessions


# ──────────────────────────────────────────────────────────────────────
# Tests — document ingestion
# ──────────────────────────────────────────────────────────────────────


class TestDocumentIngestion:
    """Tests for PDF upload and processing."""

    def setup_method(self):
        self.embedding_service = make_mock_embedding_service()
        self.vector_store = make_mock_vector_store()
        self.llm_service = make_mock_llm_service()
        self.pipeline = RAGPipeline(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            llm_service=self.llm_service,
        )

    @pytest.mark.asyncio
    async def test_ingest_document_success(self):
        sid = self.pipeline.create_session()
        pdf = make_test_pdf()

        # Make embed_texts return the right number of embeddings
        self.embedding_service.embed_texts.return_value = [[0.1] * 384] * 10

        result = await self.pipeline.ingest_document(sid, pdf, "test.pdf")

        assert isinstance(result, UploadResponse)
        assert result.success is True
        assert result.filename == "test.pdf"
        assert result.num_chunks >= 1
        assert result.num_pages >= 1
        assert "test.pdf" in self.pipeline.sessions[sid].uploaded_files

    @pytest.mark.asyncio
    async def test_ingest_invalid_session_raises(self):
        pdf = make_test_pdf()
        with pytest.raises(ValueError, match="not found"):
            await self.pipeline.ingest_document("bad_id", pdf, "test.pdf")

    @pytest.mark.asyncio
    async def test_ingest_uses_embedding_cache(self):
        sid = self.pipeline.create_session()
        pdf = make_test_pdf()

        # First ingest to determine how many chunks the PDF produces
        from pdf_processor import process_pdf
        chunks = process_pdf(pdf, "test.pdf")
        num_chunks = len(chunks)

        # Simulate cache hit with matching chunk count
        self.embedding_service.get_cached_embeddings.return_value = [
            [0.1] * 384 for _ in range(num_chunks)
        ]

        await self.pipeline.ingest_document(sid, pdf, "test.pdf")

        # embed_texts should NOT be called when cache hits
        self.embedding_service.embed_texts.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# Tests — question answering
# ──────────────────────────────────────────────────────────────────────


class TestQuestionAnswering:
    """Tests for the RAG Q&A pipeline."""

    def setup_method(self):
        self.embedding_service = make_mock_embedding_service()
        self.vector_store = make_mock_vector_store()
        self.llm_service = make_mock_llm_service()
        self.pipeline = RAGPipeline(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            llm_service=self.llm_service,
        )

    @pytest.mark.asyncio
    async def test_ask_question_happy_path(self):
        sid = self.pipeline.create_session()
        self.pipeline.sessions[sid].uploaded_files.append("test.pdf")

        result = await self.pipeline.answer_question(sid, "What is this about?")

        assert isinstance(result, AnswerResponse)
        assert result.answer == "The document discusses testing."
        assert result.confidence == 0.85
        assert len(result.sources) >= 1
        assert result.session_id == sid

    @pytest.mark.asyncio
    async def test_ask_updates_chat_history(self):
        sid = self.pipeline.create_session()
        self.pipeline.sessions[sid].uploaded_files.append("test.pdf")

        await self.pipeline.answer_question(sid, "What is this?")

        history = self.pipeline.sessions[sid].chat_history
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_ask_without_documents_raises(self):
        sid = self.pipeline.create_session()
        self.vector_store.collection_exists.return_value = False

        with pytest.raises(ValueError, match="No documents"):
            await self.pipeline.answer_question(sid, "What is this?")

    @pytest.mark.asyncio
    async def test_ask_nonexistent_session_raises(self):
        with pytest.raises(ValueError, match="not found"):
            await self.pipeline.answer_question("bad_id", "What?")

    @pytest.mark.asyncio
    async def test_followup_detection_called_with_history(self):
        sid = self.pipeline.create_session()
        session = self.pipeline.sessions[sid]
        session.uploaded_files.append("test.pdf")

        # Add some chat history
        from models import ChatMessage
        session.chat_history.append(ChatMessage(role="user", content="First question"))
        session.chat_history.append(ChatMessage(role="assistant", content="First answer"))

        await self.pipeline.answer_question(sid, "Tell me more")

        # detect_follow_up should be called when history exists
        self.llm_service.detect_follow_up.assert_called_once()

    @pytest.mark.asyncio
    async def test_followup_detection_skipped_without_history(self):
        sid = self.pipeline.create_session()
        self.pipeline.sessions[sid].uploaded_files.append("test.pdf")

        await self.pipeline.answer_question(sid, "What is this?")

        # detect_follow_up should NOT be called when no history
        self.llm_service.detect_follow_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_no_info(self):
        sid = self.pipeline.create_session()
        self.pipeline.sessions[sid].uploaded_files.append("test.pdf")
        self.vector_store.search.return_value = []

        result = await self.pipeline.answer_question(sid, "Random question")

        assert "couldn't find" in result.answer.lower()
        assert result.confidence == 0.0
        assert result.sources == []
