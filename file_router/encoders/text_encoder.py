"""Text encoder — sentence-transformers or deterministic stub."""

from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np


class TextEncoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.backend
        self.dim = cfg.dim
        self._model = None

        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                requested = os.environ.get(
                    "ENCODER_DEVICE", getattr(cfg, "device", "auto"))
                device = None if requested == "auto" else requested
                self._model = SentenceTransformer(cfg.model_name, device=device)
                self.dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:  # noqa: BLE001
                print(f"[TextEncoder] failed to load {cfg.model_name}: {e}. Falling back to stub.")
                self.backend = "stub"

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "sentence-transformers":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        return self._stub_encode(texts)

    def _stub_encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # expand the 32-byte digest into `dim` floats deterministically
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            out[i] = v
        return out
