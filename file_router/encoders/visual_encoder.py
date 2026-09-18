"""Visual encoder — ColPali (multi-vector) | CLIP (single) | stub.

The router does not care whether a node has a single-vector or multi-vector
visual embedding; it only needs `base_score` from retrieval. We hide that
difference here.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np


def _as_embedding(output) -> "np.ndarray":
    """Return the pooled projected embedding from a CLIP feature call.

    transformers <5 returned a plain tensor from get_text_features /
    get_image_features; 5.x returns a BaseModelOutputWithPooling. Indexing the
    object yields last_hidden_state (a token sequence), not the projected
    embedding, which silently produces vectors of the wrong width and space.
    Handle both shapes explicitly.
    """
    tensor = output
    if not hasattr(tensor, "shape"):
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            candidate = getattr(tensor, attr, None)
            if candidate is not None and hasattr(candidate, "shape"):
                tensor = candidate
                break
        else:
            raise TypeError(
                f"cannot extract an embedding from {type(output).__name__}; "
                "this transformers version returns an unrecognised structure")
    if tensor.dim() == 3:
        # (batch, tokens, dim) -> mean-pool, which keeps the vector comparable
        # to a stored pooled embedding rather than an arbitrary token.
        tensor = tensor.mean(dim=1)
    return tensor


class VisualEncoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.backend
        self.dim = cfg.dim
        self._model = None
        self._processor = None
        self.device = "cpu"

        if self.backend == "colpali":
            self._try_init_colpali()
        elif self.backend == "clip":
            self._try_init_clip()

    # ------------------------------------------------------------------ init
    def _try_init_clip(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            requested = os.environ.get("ENCODER_DEVICE", getattr(self.cfg, "device", "auto"))
            self.device = _resolve_device(torch, requested)
            self._model = CLIPModel.from_pretrained(self.cfg.clip_model_name)
            self._processor = CLIPProcessor.from_pretrained(self.cfg.clip_model_name)
            self._model.to(self.device).eval()
            self._torch = torch
            self.dim = int(getattr(self._model.config, "projection_dim",
                                   self.dim))
        except Exception as e:  # noqa: BLE001
            # Falling back silently produces embeddings in a different space
            # than whatever built the index, so retrieval degrades to noise
            # while the run still completes.  Keep the configured dimension so
            # the vectors at least remain shape-compatible, and make the
            # downgrade impossible to miss.
            print(f"[VisualEncoder] CLIP INIT FAILED: {e}")
            print("[VisualEncoder] falling back to the stub encoder -- visual "
                  "retrieval will be random. Results are NOT comparable to a "
                  "run with a working CLIP.")
            self.backend = "stub"
            self.dim = getattr(self.cfg, "dim", self.dim)

    def _try_init_colpali(self):
        try:
            import torch
            from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
            requested = os.environ.get("ENCODER_DEVICE", getattr(self.cfg, "device", "auto"))
            self.device = _resolve_device(torch, requested)
            self._model = ColPali.from_pretrained(self.cfg.colpali_model_name)
            self._processor = ColPaliProcessor.from_pretrained(self.cfg.colpali_model_name)
            self._model.to(self.device).eval()
            self._torch = torch
        except Exception as e:  # noqa: BLE001
            # ColPali and CLIP embed into different spaces and dimensions, so a
            # store built with one cannot be queried with the other.  The
            # fallback keeps the run alive but the index must be rebuilt.
            print(f"[VisualEncoder] ColPali INIT FAILED: {e}")
            print("[VisualEncoder] falling back to CLIP -- a store ingested "
                  "with ColPali must be rebuilt before it can be queried.")
            self.backend = "clip"
            self._try_init_clip()

    def _measure_text_dim(self) -> Optional[int]:
        """Dimension actually produced by the text tower, or None if unknown.

        transformers 5.x changed CLIPModel.get_text_features to return a
        different width than config.projection_dim, which silently breaks
        comparison against an index built under 4.x.  Measuring is the only
        reliable way to know, but a failure here must not disable CLIP -- it is
        diagnostic, not load-bearing.
        """
        if self.backend != "clip" or self._model is None:
            return None
        try:
            inputs = self._processor(text=["probe"], return_tensors="pt",
                                     padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                return int(_as_embedding(
                    self._model.get_text_features(**inputs)).shape[-1])
        except Exception as exc:  # noqa: BLE001
            print(f"[VisualEncoder] could not measure text dim: {repr(exc)[:160]}")
            return None

    def _measure_image_dim(self, image_path: str) -> Optional[int]:
        """Dimension produced by the image tower, for the same reason."""
        if self.backend != "clip" or self._model is None:
            return None
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                return int(_as_embedding(
                    self._model.get_image_features(**inputs)).shape[-1])
        except Exception as exc:  # noqa: BLE001
            print(f"[VisualEncoder] could not measure image dim: {repr(exc)[:160]}")
            return None

    # ------------------------------------------------------------------ api
    def encode_image(self, image_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (pooled_vector, multi_vector or None)."""
        if self.backend == "clip":
            return self._encode_clip(image_path), None
        if self.backend == "colpali":
            return self._encode_colpali(image_path)
        return self._stub_encode(image_path), None

    def encode_query(self, query: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Encode a text query into the SAME space as the stored visual vectors
        so we can cross-compare. Only CLIP supports true text→image; ColPali
        encodes query as a token sequence for MaxSim."""
        if self.backend == "clip":
            return self._encode_clip_text(query), None
        if self.backend == "colpali":
            return self._encode_colpali_query(query)
        return self._stub_text(query), None

    # ----------------------------------------------------------- CLIP paths
    def _encode_clip(self, image_path: str) -> np.ndarray:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = _as_embedding(self._model.get_image_features(**inputs))
        v = feats[0].cpu().numpy().astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v

    def _encode_clip_text(self, text: str) -> np.ndarray:
        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        with self._torch.no_grad():
            feats = _as_embedding(self._model.get_text_features(**inputs))
        v = feats[0].cpu().numpy().astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v

    # ---------------------------------------------------------- ColPali paths
    def _encode_colpali(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        batch = self._processor.process_images([img]).to(self._model.device)
        with self._torch.no_grad():
            mv = self._model(**batch)          # [1, n_patches, dim]
        mv = mv[0].cpu().numpy().astype(np.float32)          # (P, D)
        pooled = mv.mean(axis=0)
        pooled /= (np.linalg.norm(pooled) + 1e-9)
        return pooled, mv

    def _encode_colpali_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        batch = self._processor.process_queries([query]).to(self._model.device)
        with self._torch.no_grad():
            mv = self._model(**batch)
        mv = mv[0].cpu().numpy().astype(np.float32)
        pooled = mv.mean(axis=0)
        pooled /= (np.linalg.norm(pooled) + 1e-9)
        return pooled, mv

    # ----------------------------------------------------------- stub
    def _stub_encode(self, image_path: str) -> np.ndarray:
        with open(image_path, "rb") as f:
            h = hashlib.sha256(f.read()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v

    def _stub_text(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v

    @staticmethod
    def maxsim(query_mv: np.ndarray, doc_mv: np.ndarray) -> float:
        """ColBERT/ColPali late-interaction score."""
        sim = query_mv @ doc_mv.T            # (Q, D)
        return float(sim.max(axis=1).sum())


def _resolve_device(torch, requested: str) -> str:
    requested = str(requested or "auto").lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
