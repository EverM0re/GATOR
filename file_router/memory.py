"""Per-dataset memory facade: shared resources + retriever, built once.

Each dataset gets its own store sub-dir (store/<dataset>/) so indexes never mix.
Encoders are shared across datasets (loaded once) to save memory/time.
"""

from __future__ import annotations

import os

from .encoders import TextEncoder, VisualEncoder, VLM
from .ingestion import Ingestor
from .retrieval import Retriever
from .storage import GraphStore, NodeStore, TextIndex, VisualIndex
from .utils import info


class SharedEncoders:
    """Loaded once; reused across all datasets."""

    def __init__(self, cfg):
        self.cfg = cfg
        info("[mem] loading text encoder...")
        self.text_enc = TextEncoder(cfg.encoders.text)
        info(f"[mem] text encoder backend={self.text_enc.backend} dim={self.text_enc.dim}")
        info("[mem] loading visual encoder...")
        self.vis_enc = VisualEncoder(cfg.encoders.visual)
        info(f"[mem] visual encoder backend={self.vis_enc.backend} dim={self.vis_enc.dim}")
        # Older configs had encoders.vlm; the current config deliberately uses
        # one shared `llm` block. Keep both layouts compatible.
        self.vlm = VLM(getattr(cfg.encoders, "vlm", cfg.llm))


class DatasetMemory:
    """Storage + retriever for a single dataset, under store/<dataset>/."""

    def __init__(self, cfg, dataset: str, enc: SharedEncoders):
        self.cfg = cfg
        self.dataset = dataset
        self.enc = enc
        self.store_dir = os.path.join(cfg.paths.store_dir, dataset)
        os.makedirs(self.store_dir, exist_ok=True)
        self.nodes = NodeStore(self.store_dir)
        self.text_idx = TextIndex(self.store_dir, enc.text_enc.dim)
        self.vis_idx = VisualIndex(self.store_dir, enc.vis_enc.dim)
        self.graph = GraphStore(self.store_dir)

    def ingestor(self) -> Ingestor:
        return Ingestor(self.cfg, self.enc.text_enc, self.enc.vis_enc, self.enc.vlm,
                        self.nodes, self.text_idx, self.vis_idx, self.graph)

    def retriever(self) -> Retriever:
        return Retriever(self.cfg, self.enc.text_enc, self.enc.vis_enc,
                         self.nodes, self.text_idx, self.vis_idx, self.graph)
