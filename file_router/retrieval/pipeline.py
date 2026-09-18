"""Dual-recall retrieval: parallel text + visual → family expansion → 1-hop graph expand.

Produces a pool of `EvidenceNode`s, each annotated with a `base_score`.
GroupBuilder then wraps them into EvidenceGroup objects for the router.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..encoders import TextEncoder, VisualEncoder
from ..schemas import EvidenceGroup, EvidenceNode, ModalityBias
from ..storage import GraphStore, NodeStore, TextIndex, VisualIndex


_IMAGE_KEYWORDS = ("image", "photo", "picture", "figure", "chart", "diagram",
                   "screenshot", "visual", "看", "图", "照片", "截图")
# Keyword lists are bilingual on purpose: the corpora contain
# Chinese-language questions, so these tokens are matched against
# real input and are not translatable comments.
_PDF_KEYWORDS = ("pdf", "document", "page", "文档", "文件", "报告")


def detect_modality_bias(question: str, confidence_threshold: float) -> ModalityBias:
    q = question.lower()
    image_hits = sum(1 for kw in _IMAGE_KEYWORDS if kw in q)
    pdf_hits = sum(1 for kw in _PDF_KEYWORDS if kw in q)
    if image_hits >= pdf_hits and image_hits > 0:
        return ModalityBias("image", min(0.9, 0.5 + 0.2 * image_hits),
                            apply_bias=(0.5 + 0.2 * image_hits) >= confidence_threshold)
    if pdf_hits > 0:
        return ModalityBias("pdf", min(0.9, 0.5 + 0.2 * pdf_hits),
                            apply_bias=(0.5 + 0.2 * pdf_hits) >= confidence_threshold)
    return ModalityBias("text", 0.3, apply_bias=False)


class Retriever:
    def __init__(self, cfg,
                 text_enc: TextEncoder, vis_enc: VisualEncoder,
                 nodes: NodeStore, text_idx: TextIndex,
                 vis_idx: VisualIndex, graph: GraphStore):
        self.cfg = cfg
        self.text_enc = text_enc
        self.vis_enc = vis_enc
        self.nodes = nodes
        self.text_idx = text_idx
        self.vis_idx = vis_idx
        self.graph = graph

    # ------------------------------------------------------------ public
    def recall(self, question: str) -> Tuple[List[EvidenceGroup], ModalityBias]:
        bias = detect_modality_bias(question,
                                    self.cfg.router.modality_bias_confidence)

        # 1) text recall
        q_text_vec = self.text_enc.encode([question])[0]
        text_hits = self.text_idx.search(q_text_vec, topk=self.cfg.retrieval.topk_text)

        # 2) visual recall (only if visual index non-empty)
        visual_hits: List[Tuple[str, float]] = []
        if self.vis_idx.size() > 0:
            try:
                q_vpool, q_vmv = self.vis_enc.encode_query(question)
                visual_hits = self.vis_idx.search(
                    q_vpool, q_vmv, topk=self.cfg.retrieval.topk_visual,
                    use_maxsim=(self.cfg.encoders.visual.backend == "colpali"),
                )
            except Exception as e:  # noqa: BLE001
                # Visual recall failing on every query silently halves the
                # candidate pool, and the run still completes with plausible
                # numbers.  Warn once per process rather than once per query so
                # the message is visible instead of buried.
                if not getattr(self, "_visual_warned", False):
                    self._visual_warned = True
                    print(f"[Retriever] VISUAL RECALL DISABLED: {e}")
                    print("[Retriever] every query will run text-only from here; "
                          "results are not comparable to a run with visual recall.")

        # 3) merge → dict[node_id → score]
        merged: Dict[str, float] = {}
        fusion = getattr(self.cfg.retrieval, "fusion", "max")
        if fusion == "rrf":
            rrf_k = float(getattr(self.cfg.retrieval, "rrf_k", 60.0))
            for hits in (text_hits, visual_hits):
                for rank, (nid, _) in enumerate(hits, 1):
                    merged[nid] = merged.get(nid, 0.0) + 1.0 / (rrf_k + rank)
        else:
            for nid, score in text_hits:
                merged[nid] = max(merged.get(nid, float("-inf")), score)
            for nid, score in visual_hits:
                merged[nid] = max(merged.get(nid, float("-inf")), score)

        # Raw RRF values live in a very narrow ~0.01 range.  Feeding those
        # directly to a tiny MLP made candidate relevance almost invisible
        # compared with tier/type flags.  Normalize before graph expansion so
        # direct hits occupy [0, 1] and inherited evidence receives a decay.
        if merged:
            finite_scores = [score for score in merged.values()
                             if np.isfinite(score)]
            max_score = max(finite_scores) if finite_scores else 0.0
            if max_score > 0:
                merged = {nid: max(0.0, float(score) / max_score)
                          for nid, score in merged.items()}

        # 4) family expansion: add same-cluster multi-granularity siblings
        if self.cfg.retrieval.expand_to_family:
            merged = self._expand_family(merged)

        # 5) 1-hop graph expansion
        if self.cfg.retrieval.expand_to_neighbors:
            edge_types = ["summary_of", "part_of", "co_occurs", "linked_image"]
            decay = float(getattr(self.cfg.retrieval,
                                  "neighbor_score_decay", 0.25))
            neighbor_scores: Dict[str, float] = {}
            for nid, score in list(merged.items()):
                for neighbor in self.graph.neighbors(nid, edge_types=edge_types):
                    if neighbor not in merged:
                        neighbor_scores[neighbor] = max(
                            neighbor_scores.get(neighbor, 0.0), score * decay)
            merged.update(neighbor_scores)

        # cap
        merged = dict(sorted(merged.items(), key=lambda kv: -kv[1])
                      [:self.cfg.retrieval.max_candidates])

        # 6) wrap into EvidenceGroup
        node_map = self.nodes.get_many(list(merged.keys()))
        groups: List[EvidenceGroup] = []
        for nid, score in merged.items():
            node = node_map.get(nid)
            if node is None:
                continue
            groups.append(self._to_group(node, base_score=score))

        # 7) apply modality bias as a small additive score signal
        if bias.apply_bias:
            for g in groups:
                g.modality_bias_score = _bias_bonus(bias.active_modality,
                                                    g.node_class) * bias.confidence

        return groups, bias

    # ------------------------------------------------------------ internals
    def _expand_family(self, merged: Dict[str, float]) -> Dict[str, float]:
        """For each retrieved node, also surface its summary_of / expansion siblings.

        This gives the router multiple cost tiers for the same evidence.
        """
        to_add: Dict[str, float] = {}
        for nid, score in merged.items():
            node = self.nodes.get(nid)
            if node is None:
                continue
            for sib in (node.summary_of + node.expansions +
                        ([node.parent_summary_id] if node.parent_summary_id else [])):
                if sib and sib not in merged:
                    to_add[sib] = max(to_add.get(sib, 0.0), 0.6 * score)
        merged.update(to_add)
        return merged

    def _to_group(self, node: EvidenceNode, base_score: float) -> EvidenceGroup:
        companions = list(node.mandatory_companions)
        members = [node.node_id] + companions
        comp_cost = 0.0
        for cid in companions:
            cn = self.nodes.get(cid)
            if cn is not None:
                comp_cost += cn.token_equivalent_cost
        total_cost = node.token_equivalent_cost + comp_cost
        return EvidenceGroup(
            group_id=f"grp_{node.node_id}",
            root_node_id=node.node_id,
            node_ids=members,
            node_class=node.node_class,
            level=node.level,
            redundancy_cluster_id=node.redundancy_cluster_id,
            base_score=base_score,
            token_equivalent_cost=total_cost,
            coverage_signature=node.coverage_signature,
            source_doc_id=node.source_ref.doc_id,
        )


# ---------- modality bias table ----------
_BIAS_TABLE = {
    "image": {"image": 0.9, "pdf_page": 0.7, "pdf_region": 0.5,
              "text_span": 0.1, "text_cluster": 0.1, "doc_summary": 0.1},
    "pdf":   {"pdf_page": 0.9, "pdf_region": 0.7,
              "text_span": 0.4, "text_cluster": 0.3, "doc_summary": 0.3,
              "image": 0.3},
    "text":  {"text_span": 0.9, "text_cluster": 0.6, "doc_summary": 0.5,
              "image": 0.2, "pdf_page": 0.3, "pdf_region": 0.5},
}


def _bias_bonus(active: str, node_class: str) -> float:
    return _BIAS_TABLE.get(active, {}).get(node_class, 0.3)
