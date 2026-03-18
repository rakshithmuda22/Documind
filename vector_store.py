"""
Vector store service for DocuMind.

Wraps ChromaDB with per-session collections for document storage
and similarity search. Uses an ephemeral (in-memory) client.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import chromadb

from models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages ChromaDB collections, one per session.

    Each user session gets an isolated collection so documents
    from different users never mix.
    """

    def __init__(self) -> None:
        """Initialise the in-memory ChromaDB client."""
        self.client = chromadb.EphemeralClient()
        logger.info("VectorStoreService initialised (ephemeral/in-memory)")

    def _collection_name(self, session_id: str) -> str:
        """Build a valid ChromaDB collection name from a session ID.

        Args:
            session_id: UUID-based session identifier.

        Returns:
            Sanitised collection name (3-63 chars, alphanumeric + underscore).
        """
        safe = session_id.replace("-", "_")
        return f"s_{safe}"[:63]

    def add_documents(
        self,
        session_id: str,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Add document chunks with their embeddings to a session collection.

        Args:
            session_id: Session identifier.
            chunks: Document chunks to store.
            embeddings: Corresponding embedding vectors (same order as chunks).
        """
        collection = self.client.get_or_create_collection(
            name=self._collection_name(session_id),
            metadata={"hnsw:space": "cosine"},
        )

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_chroma_metadata() for chunk in chunks]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "Added %d chunks to session %s",
            len(chunks),
            session_id[:8],
        )

    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for the most relevant chunks in a session collection.

        Args:
            session_id: Session identifier.
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, source_filename, page_number,
            chunk_index, distance. Sorted by relevance (lowest distance first).

        Raises:
            ValueError: If the session collection does not exist.
        """
        col_name = self._collection_name(session_id)

        try:
            collection = self.client.get_collection(name=col_name)
        except Exception:
            raise ValueError(
                f"No documents found for session {session_id[:8]}. "
                "Please upload a PDF first."
            )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )

        # Unpack nested results (ChromaDB returns list-of-lists)
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        ranked = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            ranked.append({
                "text": doc,
                "source_filename": meta.get("source_filename", "unknown"),
                "page_number": meta.get("page_number", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": dist,
            })

        return ranked

    def collection_exists(self, session_id: str) -> bool:
        """Check if a session has an existing collection with documents.

        Args:
            session_id: Session identifier.

        Returns:
            True if the collection exists and has at least one document.
        """
        col_name = self._collection_name(session_id)
        try:
            collection = self.client.get_collection(name=col_name)
            return collection.count() > 0
        except Exception:
            return False

    def delete_collection(self, session_id: str) -> None:
        """Delete a session's collection and all its data.

        Args:
            session_id: Session identifier.
        """
        col_name = self._collection_name(session_id)
        try:
            self.client.delete_collection(name=col_name)
            logger.info("Deleted collection for session %s", session_id[:8])
        except Exception:
            pass  # Collection may not exist; that is fine

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about a session's stored documents.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with 'chunk_count' and 'document_names'.
        """
        col_name = self._collection_name(session_id)
        try:
            collection = self.client.get_collection(name=col_name)
            all_meta = collection.get()["metadatas"] or []
            filenames = list({
                m.get("source_filename", "unknown") for m in all_meta
            })
            return {
                "chunk_count": collection.count(),
                "document_names": filenames,
            }
        except Exception:
            return {"chunk_count": 0, "document_names": []}
