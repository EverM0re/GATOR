"""Dense text index backed by numpy (cosine on L2-normalized vectors)."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


class TextIndex:
    def __init__(self, store_dir: str, dim: int):
        self.dir = store_dir
        self.dim = dim
        self.vecs_path = os.path.join(store_dir, "text_vecs.npy")
        self.ids_path = os.path.join(store_dir, "text_ids.txt")
        self._vecs: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._ids: List[str] = []
        self._load()

    def _load(self):
        if os.path.exists(self.vecs_path) and os.path.exists(self.ids_path):
            self._vecs = np.load(self.vecs_path)
            with open(self.ids_path, "r", encoding="utf-8") as f:
                self._ids = [line.strip() for line in f if line.strip()]

    def save(self):
        np.save(self.vecs_path, self._vecs)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            for nid in self._ids:
                f.write(nid + "\n")

    def add(self, node_ids: List[str], vectors: np.ndarray):
        if vectors.shape[0] == 0:
            return
        vectors = vectors.astype(np.float32)
        # normalize in case encoder didn't
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        vectors = vectors / norms
        self._vecs = np.concatenate([self._vecs, vectors], axis=0) if self._vecs.size else vectors
        self._ids.extend(node_ids)

    def search(self, query_vec: np.ndarray, topk: int = 20) -> List[Tuple[str, float]]:
        if self._vecs.shape[0] == 0:
            return []
        q = query_vec.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._vecs @ q                  # (N,)
        idx = np.argsort(-scores)[:topk]
        return [(self._ids[i], float(scores[i])) for i in idx]

    def size(self) -> int:
        return len(self._ids)
