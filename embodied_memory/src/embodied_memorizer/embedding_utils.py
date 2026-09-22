"""
Embedding utilities for memory retrieval.

Supports sentence-transformers for text embedding and
optional CLIP for image embedding.
"""

import hashlib
import re
import numpy as np
from typing import List, Optional, Union


class EmbeddingEngine:
    """
    Computes text (and optionally image) embeddings.

    Two backends:
    - Remote (vLLM / OpenAI-compatible server): pass ``base_url``.
      The server must expose ``POST /v1/embeddings``.
    - Local (sentence-transformers): leave ``base_url=None``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda:1",
        base_url: Optional[str] = None,
        api_key: str = "dummy",
    ):
        self.model_name = model_name
        self.device = device
        self.base_url = base_url
        self.api_key = api_key
        self.hash_dim = 384
        self._text_model = None  # sentence-transformers (local only)
        self._client = None  # openai.OpenAI (remote only)
        self._clip_model = None
        self._clip_processor = None

    # ── 后端初始化 ────────────────────────────────────────────────────────────

    def _load_remote_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _load_text_model(self):
        if self._text_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embedding. "
                "Install with: pip install sentence-transformers"
            )

    # ── 编码 ─────────────────────────────────────────────────────────────────

    def encode_text(
        self, texts: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """Encode text(s) to embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]
        if self._use_lexical_hash():
            return self._encode_lexical_hash(texts, normalize)
        if self.base_url:
            return self._encode_remote(texts, normalize)
        return self._encode_local(texts, normalize)

    def _encode_remote(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Call vLLM / OpenAI-compatible /v1/embeddings endpoint."""
        self._load_remote_client()
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        embeddings = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms
        return embeddings

    def _encode_local(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Encode with local sentence-transformers model."""
        self._load_text_model()
        return self._text_model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

    def _use_lexical_hash(self) -> bool:
        """Return True for the explicit offline lexical-hash backend.

        This keeps embodied_memorizer usable on machines without a cached
        sentence-transformers model or external network access. It is a generic
        text retrieval backend, not a task-specific answer source.
        """
        name = (self.model_name or "").lower()
        return name in {"lexical-hash", "hash", "offline-hash"}

    def _tokenize(self, text: str) -> List[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }
        words = [
            w
            for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in stopwords
        ]
        tokens = []
        for word in words:
            tokens.append(word)
            if len(word) > 3 and word.endswith("ies"):
                tokens.append(word[:-3] + "y")
            elif len(word) > 3 and word.endswith("es"):
                tokens.append(word[:-1])
                tokens.append(word[:-2])
            elif len(word) > 3 and word.endswith("s"):
                tokens.append(word[:-1])
        tokens.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        return tokens

    def _encode_lexical_hash(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Encode text with a deterministic hashed bag-of-words vector."""
        vectors = np.zeros((len(texts), self.hash_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.hash_dim
                vectors[row, idx] += 1.0
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms
        return vectors

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text, return 1D vector."""
        return self.encode_text(text)[0]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(
        query: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and multiple candidates."""
        if len(candidates) == 0:
            return np.array([])
        norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = candidates / norms
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(len(candidates))
        normed_q = query / q_norm
        return normed @ normed_q
