"""
Ollama LLM Helper — local language model integration.

Auto-detects Ollama availability and provides a context-aware answer
generator. Falls back gracefully when Ollama is not installed.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import List, Optional

logger = logging.getLogger(__name__)

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
DEFAULT_MODEL = "llama3"
SYSTEM_PROMPT = (
    "You are the IOCL Knowledge Assistant. Answer the user's question ONLY "
    "using the provided context. If the context does not contain enough "
    'information, reply with "Information not found in the database." '
    "Do not use any external knowledge."
)


class OllamaLLM:
    """Interface to a locally running Ollama LLM server."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._available: Optional[bool] = None  # lazily detected

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Check if Ollama is reachable on localhost."""
        if self._available is not None:
            return self._available
        try:
            sock = socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=3)
            sock.close()
            self._available = True
            logger.info("Ollama detected at %s:%d", OLLAMA_HOST, OLLAMA_PORT)
        except (socket.timeout, ConnectionRefusedError, OSError):
            self._available = False
            logger.info("Ollama not available — falling back to context retrieval.")
        return self._available

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        query: str,
        context_docs: List[str],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate an answer using local Ollama LLM, or fall back gracefully."""
        if not self.is_available():
            return self._fallback_answer(context_docs)

        context_block = "\n\n---\n\n".join(context_docs) if context_docs else ""
        prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Answer based ONLY on the context above:"
        )

        try:
            import urllib.request  # noqa: WPS433 — stdlib, lazy import

            payload = json.dumps({
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }).encode()

            req = urllib.request.Request(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())

            answer = data.get("message", {}).get("content", "")
            if answer.strip():
                return answer.strip()
            return self._fallback_answer(context_docs)

        except Exception:
            logger.exception("Ollama generation failed")
            return self._fallback_answer(context_docs)

    # ------------------------------------------------------------------
    # Fallback (no Ollama)
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_answer(context_docs: List[str]) -> str:
        """Return the most relevant retrieved context as the answer."""
        if not context_docs:
            return "Information not found in the database."
        return context_docs[0]
