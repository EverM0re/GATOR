"""Ingest a unified corpus.jsonl into the 3-tier cost ladder (DESIGN.md §2).

For every page we emit up to THREE EvidenceNodes that share one
`redundancy_cluster_id = f"{doc_id}:p{page_num}"`, so the router's family_mutex
forces it to pick at most ONE cost tier per page:

    caption tier   node_class=doc_summary  level=1   cheapest   (caption text)
    screenshot     node_class=pdf_page      level=0   mid        (page image)
    full-text      node_class=text_span     level=0   priciest   (full page text)

Caption is built offline (heuristic: first sentence/heading) unless
ingestion.caption.backend == "vlm".

Cost is baked here (token-equivalent), never recomputed downstream.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import List, Optional

import numpy as np

from ..encoders import TextEncoder, VisualEncoder, VLM
from ..schemas import EvidenceNode, SourceRef
from ..storage import GraphStore, NodeStore, TextIndex, VisualIndex
from ..utils import dbg, info, warn


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _cluster_id(doc_id: str, page_num: int) -> str:
    return f"{doc_id}:p{page_num}"


def _coverage_sig(vec: Optional[np.ndarray]) -> Optional[List[float]]:
    if vec is None:
        return None
    v = np.asarray(vec, dtype=np.float32)[:64]
    if v.size == 0:
        return None
    if v.size < 64:
        v = np.pad(v, (0, 64 - v.size))
    n = np.linalg.norm(v) + 1e-9
    return (v / n).tolist()


def _heuristic_caption(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # first sentence or first line, capped
    for sep in (". ", "\n"):
        if sep in text:
            cand = text.split(sep)[0].strip()
            if 10 <= len(cand) <= max_chars:
                return cand
    return text[:max_chars]


class Ingestor:
    def __init__(self, cfg, text_enc, vis_enc, vlm, nodes, text_idx, vis_idx, graph):
        self.cfg = cfg
        self.text_enc = text_enc
        self.vis_enc = vis_enc
        self.vlm = vlm
        self.nodes = nodes
        self.text_idx = text_idx
        self.vis_idx = vis_idx
        self.graph = graph
        self._ocr_warned = False
        self._caption_cache_path = os.path.join(
            self.cfg.paths.data_dir, "vlm_caption_cache.json")
        self._caption_cache = {}
        if os.path.exists(self._caption_cache_path):
            try:
                with open(self._caption_cache_path, encoding="utf-8") as cache_file:
                    self._caption_cache = json.load(cache_file)
            except Exception:  # noqa: BLE001
                self._caption_cache = {}

    # ------------------------------------------------------------ cost model
    def _text_cost(self, text: str) -> float:
        return len(text or "") / self.cfg.cost.text_chars_per_token

    def _screenshot_cost(self) -> float:
        return self.cfg.cost.pdf_page_token_budget * self.cfg.cost.visual_token_ratio

    # ------------------------------------------------------------ main
    def ingest_corpus(self, corpus_path: str) -> int:
        if not os.path.exists(corpus_path):
            warn(f"[ingest] no corpus at {corpus_path}"); return 0
        n_nodes = 0
        text_batch_ids, text_batch_vecs = [], []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                added = self._ingest_doc(doc, text_batch_ids, text_batch_vecs)
                n_nodes += added
        # flush text index
        if text_batch_vecs:
            self.text_idx.add(text_batch_ids, np.vstack(text_batch_vecs))
        self.text_idx.save()
        self.vis_idx.save()
        self.graph.save()
        info(f"[ingest] {corpus_path}: {n_nodes} nodes "
             f"(text_idx={self.text_idx.size()}, visual_idx={self.vis_idx.size()})")
        return n_nodes

    def _ingest_doc(self, doc, text_batch_ids, text_batch_vecs) -> int:
        doc_id = doc["doc_id"]
        added = 0
        for page in doc.get("pages", []):
            pnum = page.get("page_num", 0)
            cluster = _cluster_id(doc_id, pnum)
            full_text = (page.get("text") or "").strip()
            img_path = page.get("image_path") or ""
            caption = (page.get("caption") or "").strip()
            if (not full_text and img_path and
                    bool(getattr(getattr(self.cfg.ingestion, "ocr", None),
                                 "enabled", False))):
                full_text = self._ocr_image(img_path)

            tier_ids: List[str] = []   # to wire summary_of edges among tiers

            # ---- caption tier (L1, cheapest) ----
            if not caption:
                dataset = doc.get("dataset")
                caption_cfg = self.cfg.ingestion.caption
                use_vlm = (
                    caption_cfg.backend == "vlm" or
                    (bool(getattr(caption_cfg, "vlm_for_image_only", False)) and
                     not full_text and dataset in set(self.cfg.datasets.enabled))
                )
                if use_vlm and img_path and self.vlm.backend != "stub":
                    try:
                        cache_key = hashlib.sha256(
                            (os.path.abspath(img_path) + "|" +
                             self.vlm._model_name()).encode("utf-8")
                        ).hexdigest()
                        caption = self._caption_cache.get(cache_key, "")
                        if not caption:
                            generated = self.vlm.summarize_page(
                                img_path, full_text[:1500])
                            if (self.vlm.last_call_ok and
                                    not self.vlm.last_call_was_stub):
                                caption = generated
                                self._caption_cache[cache_key] = caption
                                os.makedirs(os.path.dirname(
                                    self._caption_cache_path), exist_ok=True)
                                with open(self._caption_cache_path, "w",
                                          encoding="utf-8") as cache_file:
                                    json.dump(self._caption_cache, cache_file,
                                              ensure_ascii=False, indent=2)
                    except Exception as e:  # noqa: BLE001
                        warn(f"[ingest] vlm caption failed: {repr(e)[:60]}")
                if not caption:
                    caption = _heuristic_caption(full_text,
                                                 self.cfg.ingestion.caption.heuristic_max_chars)
            if caption:
                nid = f"{doc_id}_p{pnum}_cap"
                vec = self.text_enc.encode([caption])[0]
                node = EvidenceNode(
                    node_id=nid, level=1, node_class="doc_summary",
                    source_ref=SourceRef(doc_id=doc_id, doc_type=doc.get("doc_type", "pdf"),
                                         page_num=pnum),
                    text=caption,
                    text_embedding=vec.tolist(),
                    redundancy_cluster_id=cluster,
                    coverage_signature=_coverage_sig(vec),
                    text_token_cost=self._text_cost(caption),
                    token_equivalent_cost=self._text_cost(caption),
                    created_at=_now(),
                    metadata={"tier": "caption", "dataset": doc.get("dataset")},
                )
                self.nodes.put(node)
                text_batch_ids.append(nid); text_batch_vecs.append(vec[None, :])
                self.graph.add_node(nid, level=1)
                tier_ids.append(nid); added += 1
                dbg(f"[ingest] {nid} caption cost={node.token_equivalent_cost:.0f}")

            # ---- screenshot tier (L0 visual, mid) ----
            if img_path and os.path.exists(img_path) and self.cfg.ingestion.pdf.render_page_images:
                nid = f"{doc_id}_p{pnum}_img"
                try:
                    pooled, mv = self.vis_enc.encode_image(img_path)
                    node = EvidenceNode(
                        node_id=nid, level=0, node_class="pdf_page",
                        source_ref=SourceRef(doc_id=doc_id, doc_type=doc.get("doc_type", "pdf"),
                                             page_num=pnum),
                        page_image_path=img_path,
                        visual_embedding=pooled.tolist(),
                        visual_multi_vector=(mv.tolist() if mv is not None else None),
                        redundancy_cluster_id=cluster,
                        coverage_signature=_coverage_sig(pooled),
                        visual_token_cost=self._screenshot_cost(),
                        token_equivalent_cost=self._screenshot_cost(),
                        created_at=_now(),
                        metadata={"tier": "screenshot", "dataset": doc.get("dataset")},
                    )
                    self.nodes.put(node)
                    self.vis_idx.add(nid, pooled, mv)
                    self.graph.add_node(nid, level=0)
                    tier_ids.append(nid); added += 1
                    dbg(f"[ingest] {nid} screenshot cost={node.token_equivalent_cost:.0f}")
                except Exception as e:  # noqa: BLE001
                    warn(f"[ingest] screenshot encode failed for {img_path}: {repr(e)[:60]}")

            # ---- full-text tier (L0 text, priciest when long) ----
            if full_text:
                nid = f"{doc_id}_p{pnum}_txt"
                vec = self.text_enc.encode([full_text[:4000]])[0]
                node = EvidenceNode(
                    node_id=nid, level=0, node_class="text_span",
                    source_ref=SourceRef(doc_id=doc_id, doc_type=doc.get("doc_type", "pdf"),
                                         page_num=pnum),
                    text=full_text,
                    text_embedding=vec.tolist(),
                    redundancy_cluster_id=cluster,
                    coverage_signature=_coverage_sig(vec),
                    text_token_cost=self._text_cost(full_text),
                    token_equivalent_cost=self._text_cost(full_text),
                    created_at=_now(),
                    metadata={"tier": "fulltext", "dataset": doc.get("dataset")},
                )
                self.nodes.put(node)
                text_batch_ids.append(nid); text_batch_vecs.append(vec[None, :])
                self.graph.add_node(nid, level=0)
                tier_ids.append(nid); added += 1
                dbg(f"[ingest] {nid} fulltext cost={node.token_equivalent_cost:.0f}")

            # ---- wire the tiers together so family expansion surfaces all of them ----
            # caption summarizes the heavier tiers; heavier tiers are expansions of caption
            cap_ids = [x for x in tier_ids if x.endswith("_cap")]
            heavy_ids = [x for x in tier_ids if not x.endswith("_cap")]
            for cap in cap_ids:
                cn = self.nodes.get(cap)
                cn.expansions = heavy_ids
                self.nodes.put(cn)
                for h in heavy_ids:
                    self.graph.add_edge(cap, h, edge_type="summary_of")
                    hn = self.nodes.get(h)
                    hn.summary_of = [cap]
                    hn.parent_summary_id = cap
                    # A screenshot request should also carry its cached cheap
                    # caption.  This improves grounding and provides usable
                    # text if a multimodal server has to retry an image call.
                    if h.endswith("_img") and cap not in hn.mandatory_companions:
                        hn.mandatory_companions.append(cap)
                    self.nodes.put(hn)
            # heavy tiers co-occur with each other
            for a in heavy_ids:
                for b in heavy_ids:
                    if a != b:
                        self.graph.add_edge(a, b, edge_type="co_occurs")
        return added

    def _ocr_image(self, image_path: str) -> str:
        """Optional OCR. Missing system dependencies degrade safely to no OCR."""
        try:
            import pytesseract
            from PIL import Image
            return (pytesseract.image_to_string(
                Image.open(image_path).convert("RGB")) or "").strip()
        except Exception as exc:  # noqa: BLE001
            if not self._ocr_warned:
                warn(f"[ingest] OCR unavailable; continuing without it: {repr(exc)[:120]}")
                self._ocr_warned = True
            return ""
