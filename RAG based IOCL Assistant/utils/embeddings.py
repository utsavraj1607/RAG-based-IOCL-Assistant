"""
Embeddings Engine — Sentence Transformers wrapper.

Provides text-to-vector conversion using the all-MiniLM-L6-v2 model.
Designed for fully offline operation with zero API dependencies.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingEngine:
    """Thin wrapper around Sentence Transformers for generating text embeddings."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Lazy-load the model so the first UI interaction is not blocked.
    # ------------------------------------------------------------------
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading Sentence Transformers model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Model loaded. Embedding dimension: %d", self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSION

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """Return a single embedding vector for *text*."""
        return self.model.encode(text, show_progress_bar=False, normalize_embeddings=True)

    def embed_batch(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Return an (N, dim) matrix of embeddings for *texts*."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Module-level singleton — imported by other utils & app.py
# ---------------------------------------------------------------------------
_default_engine: EmbeddingEngine | None = None


def get_embedding_engine() -> EmbeddingEngine:
    """Return the shared EmbeddingEngine instance (created on first call)."""
    global _default_engine
    if _default_engine is None:
        _default_engine = EmbeddingEngine()
    return _default_engine
