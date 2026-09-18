"""Visual index — single-vector (CLIP) and optional multi-vector (ColPali MaxSim)."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np


class VisualIndex:
    def __init__(self, store_dir: str, dim: int):
        self.dir = store_dir
        self.dim = dim
        self.pooled_path = os.path.join(store_dir, "visual_pooled.npy")
        self.ids_path = os.path.join(store_dir, "visual_ids.txt")
        self.mv_path = os.path.join(store_dir, "visual_mv.pkl")
        self._pooled: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._ids: List[str] = []
        self._mv: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.pooled_path) and os.path.exists(self.ids_path):
            self._pooled = np.load(self.pooled_path)
            with open(self.ids_path, "r", encoding="utf-8") as f:
                self._ids = [line.strip() for line in f if line.strip()]
        if os.path.exists(self.mv_path):
            with open(self.mv_path, "rb") as f:
                self._mv = pickle.load(f)

    def save(self):
        np.save(self.pooled_path, self._pooled)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            for nid in self._ids:
                f.write(nid + "\n")
        if self._mv:
            with open(self.mv_path, "wb") as f:
                pickle.dump(self._mv, f)

    def add(self, node_id: str, pooled: np.ndarray, multi_vec: Optional[np.ndarray] = None):
        p = pooled.astype(np.float32).reshape(-1)
        # Catch a dimension change at write time, where the fix is obvious,
        # rather than at query time in a later run where it is not.
        if self._pooled.size and p.shape[0] != self._pooled.shape[1]:
            raise ValueError(
                f"visual encoder changed dimension mid-ingest: index holds "
                f"{self._pooled.shape[1]}-d vectors but {node_id} is "
                f"{p.shape[0]}-d. Delete {self.pooled_path} and re-ingest.")
        p = p / (np.linalg.norm(p) + 1e-9)
        self._pooled = np.concatenate([self._pooled, p[None, :]], axis=0) if self._pooled.size else p[None, :]
        self._ids.append(node_id)
        if multi_vec is not None:
            self._mv[node_id] = multi_vec.astype(np.float32)

    def search(self, query_pooled: np.ndarray,
               query_mv: Optional[np.ndarray] = None,
               topk: int = 10, use_maxsim: bool = False) -> List[Tuple[str, float]]:
        if self._pooled.shape[0] == 0:
            return []
        q = query_pooled.astype(np.float32).reshape(-1)
        # A stored index and a query encoder that disagree on dimension means
        # the index was built by a different encoder than the one now running.
        # The matmul would fail with an opaque gufunc error that the caller
        # swallows, silently disabling visual retrieval for the whole run, so
        # say exactly what mismatched instead.
        if q.shape[0] != self._pooled.shape[1]:
            raise ValueError(
                f"visual index dimension mismatch: index is "
                f"{self._pooled.shape[1]}-d ({self._pooled.shape[0]} vectors in "
                f"{self.dir}) but the query encoder produced {q.shape[0]}-d. "
                f"The store was built with a different visual encoder; re-run "
                f"ingest with SKIP_INGEST=0 to rebuild it.")
        q = q / (np.linalg.norm(q) + 1e-9)
        pooled_scores = self._pooled @ q
        if use_maxsim and query_mv is not None and self._mv:
            # Rerank top-3×topk by MaxSim
            cand_idx = np.argsort(-pooled_scores)[:max(topk * 3, topk)]
            rescored = []
            for i in cand_idx:
                nid = self._ids[i]
                doc_mv = self._mv.get(nid)
                if doc_mv is None:
                    rescored.append((nid, float(pooled_scores[i])))
                else:
                    sim = query_mv @ doc_mv.T
                    rescored.append((nid, float(sim.max(axis=1).sum())))
            rescored.sort(key=lambda x: -x[1])
            return rescored[:topk]
        idx = np.argsort(-pooled_scores)[:topk]
        return [(self._ids[i], float(pooled_scores[i])) for i in idx]

    def size(self) -> int:
        return len(self._ids)
