"""Sentence embeddings for semantic header matching (FR-2 pass 2).

Primary: sentence-transformers (open-source, runs locally on CPU).
    EMBED_MODEL default "sentence-transformers/all-MiniLM-L6-v2"
    (BAAI/bge-small-en-v1.5 or intfloat/e5-small-v2 also work).
Fallback: character tri-gram vectors, so the system still runs without torch.
"""
from __future__ import annotations

import logging
import os
import zlib
from collections import Counter
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None
        self.backend = "trigram"
        if os.getenv("EMBED_DISABLED", "0") != "1":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name)
                self.backend = "sentence-transformers"
            except Exception as exc:  # not installed / no model download
                log.warning("sentence-transformers unavailable (%s); using tri-gram fallback", exc)

    # Semantic threshold differs by backend because score scales differ.
    @property
    def threshold(self) -> float:
        return 0.55 if self.backend == "sentence-transformers" else 0.50

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            v = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        return np.vstack([_trigram_vec(t) for t in texts]) if texts else np.zeros((0, _DIM), np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a: (n,d) b: (m,d) -> (n,m). Inputs are already L2-normalised."""
        if a.size == 0 or b.size == 0:
            return np.zeros((len(a), len(b)), np.float32)
        return a @ b.T


_DIM = 4096


@lru_cache(maxsize=4096)
def _trigram_vec(text: str) -> np.ndarray:
    s = f"  {text.lower()}  "
    grams = Counter(s[i:i + 3] for i in range(len(s) - 2))
    v = np.zeros(_DIM, dtype=np.float32)
    for g, c in grams.items():
        v[zlib.crc32(g.encode()) % _DIM] += c  # stable across processes
    n = np.linalg.norm(v)
    return v / n if n else v
