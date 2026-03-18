"""Tests for vector_store module."""

from __future__ import annotations

import random

import pytest

from models import DocumentChunk
from vector_store import VectorStoreService

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384


def fake_embedding(seed: int = 0) -> list[float]:
    """Generate a deterministic fake 384-dim embedding."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(EMBEDDING_DIM)]


def make_chunk(index: int, text: str = "chunk text", filename: str = "test.pdf") -> DocumentChunk:
    """Create a test DocumentChunk."""
    return DocumentChunk(
        id=f"hash_{index}",
        text=text,
        page_number=1,
        source_filename=filename,
        chunk_index=index,
    )


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestVectorStore:
    """Tests for VectorStoreService."""

    def setup_method(self):
        """Create a fresh store for each test."""
        self.store = VectorStoreService()

    def test_add_and_query(self):
        """Adding documents then querying should return results."""
        chunks = [make_chunk(i, f"chunk {i}") for i in range(3)]
        embeddings = [fake_embedding(i) for i in range(3)]

        self.store.add_documents("session1", chunks, embeddings)

        results = self.store.search("session1", fake_embedding(0), top_k=2)
        assert len(results) == 2
        assert "text" in results[0]
        assert "source_filename" in results[0]
        assert "distance" in results[0]

    def test_query_returns_ranked_results(self):
        """Results should be ordered by similarity (distance)."""
        chunks = [make_chunk(i, f"chunk {i}") for i in range(5)]
        embeddings = [fake_embedding(i) for i in range(5)]

        self.store.add_documents("session1", chunks, embeddings)

        # Query with embedding similar to chunk 0
        results = self.store.search("session1", fake_embedding(0), top_k=5)
        assert len(results) == 5
        # First result should be the most similar (lowest distance)
        assert results[0]["distance"] <= results[1]["distance"]

    def test_collection_exists(self):
        """collection_exists should reflect stored data."""
        assert not self.store.collection_exists("empty_session")

        chunks = [make_chunk(0)]
        embeddings = [fake_embedding(0)]
        self.store.add_documents("session1", chunks, embeddings)

        assert self.store.collection_exists("session1")

    def test_delete_collection(self):
        """Deleting a collection should remove all data."""
        chunks = [make_chunk(0)]
        embeddings = [fake_embedding(0)]
        self.store.add_documents("session1", chunks, embeddings)

        assert self.store.collection_exists("session1")
        self.store.delete_collection("session1")
        assert not self.store.collection_exists("session1")

    def test_delete_nonexistent_collection(self):
        """Deleting a non-existent collection should not raise."""
        self.store.delete_collection("does_not_exist")  # Should not raise

    def test_session_isolation(self):
        """Documents in session A should not appear in session B."""
        chunk_a = [make_chunk(0, "session A data")]
        chunk_b = [make_chunk(1, "session B data")]

        self.store.add_documents("session_a", chunk_a, [fake_embedding(10)])
        self.store.add_documents("session_b", chunk_b, [fake_embedding(20)])

        results_a = self.store.search("session_a", fake_embedding(10), top_k=5)
        results_b = self.store.search("session_b", fake_embedding(20), top_k=5)

        assert len(results_a) == 1
        assert len(results_b) == 1
        assert "session A" in results_a[0]["text"]
        assert "session B" in results_b[0]["text"]

    def test_query_nonexistent_session_raises(self):
        """Querying a non-existent session should raise ValueError."""
        with pytest.raises(ValueError, match="No documents found"):
            self.store.search("no_session", fake_embedding(0))

    def test_get_session_stats(self):
        """Session stats should reflect stored data."""
        chunks = [
            make_chunk(0, "text", "doc1.pdf"),
            make_chunk(1, "text", "doc2.pdf"),
        ]
        embeddings = [fake_embedding(0), fake_embedding(1)]
        self.store.add_documents("session1", chunks, embeddings)

        stats = self.store.get_session_stats("session1")
        assert stats["chunk_count"] == 2
        assert set(stats["document_names"]) == {"doc1.pdf", "doc2.pdf"}

    def test_get_session_stats_empty(self):
        """Stats for non-existent session should return zeros."""
        stats = self.store.get_session_stats("no_session")
        assert stats["chunk_count"] == 0
        assert stats["document_names"] == []
