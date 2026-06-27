"""
Semantic Retriever — FAISS-backed vector search engine.

Handles index construction, persistence, and multi-strategy retrieval
(keyword, semantic, hybrid) with relevance scoring.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from utils.embeddings import EmbeddingEngine, get_embedding_engine

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

INDEX_PATH = CACHE_DIR / "faiss.index"
META_PATH = CACHE_DIR / "metadata.pkl"


class SemanticRetriever:
    """FAISS-based retriever with hybrid search and relevance scoring."""

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None) -> None:
        self._engine = embedding_engine or get_embedding_engine()
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: List[Dict[str, Any]] = []
        self._searchable_texts: List[str] = []
        self._query_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    @property
    def query_log(self) -> List[Dict[str, Any]]:
        return list(self._query_log)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    @staticmethod
    def build_searchable_text(row: pd.Series) -> str:
        """Convert a DataFrame row into a single searchable string."""
        parts: List[str] = []
        for col in row.index:
            val = str(row[col]).strip()
            if val and val.lower() not in ("nan", "none", ""):
                parts.append(val)
        return " ".join(parts)

    @staticmethod
    def _detect_id_column(columns: pd.Index) -> Optional[str]:
        candidates = ["fact_id", "id", "record_id", "serial", "sr_no", "index"]
        for col in columns:
            if col.lower().strip() in candidates:
                return col
        return None

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build_index(self, df: pd.DataFrame, *, force_rebuild: bool = False) -> None:
        """Build the FAISS index from a pandas DataFrame."""
        if df.empty:
            logger.warning("DataFrame is empty — cannot build index.")
            return

        id_col = self._detect_id_column(df.columns)
        self._metadata = []
        self._searchable_texts = []

        for idx, row in df.iterrows():
            searchable = self.build_searchable_text(row)
            self._searchable_texts.append(searchable)

            meta: Dict[str, Any] = {"row_index": idx}
            for col in df.columns:
                meta[col] = row[col] if pd.notna(row[col]) else ""
            if id_col:
                meta["fact_id"] = row[id_col]
            self._metadata.append(meta)

        logger.info("Encoding %d texts …", len(self._searchable_texts))
        embeddings = self._engine.embed_batch(self._searchable_texts)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner-product (cosine on normalized vectors)
        self._index.add(embeddings)
        logger.info("FAISS index built with %d vectors (dim=%d).", self._index.ntotal, dim)

        self._save_cache()

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------
    def _save_cache(self) -> None:
        try:
            if self._index is not None:
                faiss.write_index(self._index, str(INDEX_PATH))
            with open(META_PATH, "wb") as fh:
                pickle.dump(
                    {"metadata": self._metadata, "texts": self._searchable_texts},
                    fh,
                )
            logger.info("Index cache saved to %s", CACHE_DIR)
        except Exception:
            logger.exception("Failed to save index cache.")

    def load_cache(self) -> bool:
        """Try to restore a previously persisted index. Returns True on success."""
        if not INDEX_PATH.exists() or not META_PATH.exists():
            return False
        try:
            self._index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as fh:
                data = pickle.load(fh)
            self._metadata = data["metadata"]
            self._searchable_texts = data["texts"]
            logger.info("Loaded cached index (%d vectors).", self._index.ntotal)
            return True
        except Exception:
            logger.exception("Failed to load cached index.")
            return False

    def rebuild_index(self, df: pd.DataFrame) -> None:
        """Delete cached index and rebuild from scratch."""
        for path in (INDEX_PATH, META_PATH):
            if path.exists():
                path.unlink()
        self.build_index(df, force_rebuild=True)

    # ------------------------------------------------------------------
    # Retrieval strategies
    # ------------------------------------------------------------------
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Pure semantic search via FAISS."""
        if not self.is_ready:
            return []

        q_emb = self._engine.embed_text(query).reshape(1, -1)
        scores, indices = self._index.search(q_emb, min(top_k * 2, self._index.ntotal))

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < threshold:
                continue
            entry = dict(self._metadata[idx])
            entry["_score"] = float(score)
            entry["_source"] = "semantic"
            results.append(entry)
            if len(results) >= top_k:
                break

        self._log_query(query, results)
        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Simple keyword matching across searchable texts."""
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        if not keywords:
            return []

        scored: List[Tuple[float, int]] = []
        for i, text in enumerate(self._searchable_texts):
            lower = text.lower()
            hits = sum(1 for kw in keywords if kw in lower)
            if hits > 0:
                scored.append((hits / len(keywords), i))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for score, idx in scored[:top_k]:
            entry = dict(self._metadata[idx])
            entry["_score"] = float(score)
            entry["_source"] = "keyword"
            results.append(entry)
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Combine semantic + keyword scores for a more robust retrieval."""
        sem_results = self.semantic_search(query, top_k=top_k * 2, threshold=threshold)
        kw_results = self.keyword_search(query, top_k=top_k * 2)

        scores_map: Dict[int, float] = {}
        meta_map: Dict[int, Dict[str, Any]] = {}

        for r in sem_results:
            key = hash(str(r))
            scores_map[key] = scores_map.get(key, 0) + r["_score"] * semantic_weight
            meta_map[key] = r

        for r in kw_results:
            key = hash(str(r))
            scores_map[key] = scores_map.get(key, 0) + r["_score"] * keyword_weight
            if key not in meta_map:
                meta_map[key] = r

        merged = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for key, combined_score in merged[:top_k]:
            entry = meta_map[key]
            entry["_score"] = combined_score
            entry["_source"] = "hybrid"
            results.append(entry)

        self._log_query(query, results)
        return results

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------
    @staticmethod
    def expand_query(query: str) -> str:
        """Lightweight heuristic query expansion — no external API."""
        synonyms_map: Dict[str, List[str]] = {
            "location": ["situated", "located", "place"],
            "capacity": ["volume", "output", "production"],
            "products": ["items", "goods", "output"],
            "commissioned": ["established", "started", "began"],
            "safety": ["protection", "precaution", "security"],
            "pipeline": ["transmission", "transport", "conduit"],
            "fuel": ["petrol", "diesel", "energy"],
        }
        extra_words: List[str] = []
        for token in query.lower().split():
            if token in synonyms_map:
                extra_words.extend(synonyms_map[token])
        if extra_words:
            return query + " " + " ".join(extra_words)
        return query

    # ------------------------------------------------------------------
    # Query logging
    # ------------------------------------------------------------------
    def _log_query(self, query: str, results: List[Dict[str, Any]]) -> None:
        self._query_log.append({
            "query": query,
            "results_count": len(results),
            "top_score": results[0]["_score"] if results else 0.0,
        })

    def get_category_stats(self) -> Dict[str, int]:
        """Return {category: count} from indexed metadata."""
        stats: Dict[str, int] = {}
        for m in self._metadata:
            cat = m.get("Category", "Unknown")
            stats[cat] = stats.get(cat, 0) + 1
        return stats

    def get_topic_stats(self) -> Dict[str, int]:
        """Return {topic: count} from indexed metadata."""
        stats: Dict[str, int] = {}
        for m in self._metadata:
            topic = m.get("Topic", "Unknown")
            stats[topic] = stats.get(topic, 0) + 1
        return stats

    def get_keyword_freq(self) -> Dict[str, int]:
        """Compute keyword frequency from the Keywords column."""
        freq: Dict[str, int] = {}
        for m in self._metadata:
            kw_str = m.get("Keywords", "")
            for kw in str(kw_str).split():
                kw = kw.strip().lower()
                if kw:
                    freq[kw] = freq.get(kw, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:30])