"""
Embedding service for DocuMind.

Wraps sentence-transformers to provide text embedding with caching.
The model (all-MiniLM-L6-v2, 384-dim) is loaded once at startup
and reused for all requests.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
BATCH_SIZE: int = 32


class EmbeddingService:
    """Manages the sentence-transformer model and embedding cache.

    Attributes:
        model_name: Name of the sentence-transformer model.
        model: Loaded SentenceTransformer instance.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Load the embedding model.

        Args:
            model_name: HuggingFace model identifier.
        """
        logger.info("Loading embedding model '%s'... (this may take 30s on first run)", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, List[List[float]]] = {}
        logger.info("Embedding model loaded — dimension: %d", EMBEDDING_DIM)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each 384-dim float list).
        """
        if not texts:
            return []

        logger.info("Embedding %d text(s)...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.

        Args:
            query: The search query.

        Returns:
            384-dim embedding vector as a list of floats.
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def get_cached_embeddings(self, doc_hash: str) -> Optional[List[List[float]]]:
        """Retrieve cached embeddings for a document.

        Args:
            doc_hash: Hash of the document bytes.

        Returns:
            Cached embeddings or None if not found.
        """
        return self._cache.get(doc_hash)

    def cache_embeddings(self, doc_hash: str, embeddings: List[List[float]]) -> None:
        """Store embeddings in the in-memory cache.

        Args:
            doc_hash: Hash of the document bytes.
            embeddings: List of embedding vectors to cache.
        """
        self._cache[doc_hash] = embeddings
        logger.info("Cached embeddings for doc %s (%d vectors)", doc_hash[:8], len(embeddings))
