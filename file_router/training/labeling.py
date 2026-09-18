"""Auto-generate gold / minimal-sufficient labels (DESIGN.md §3.1).

Replaces cnem_v's manual `fill_gold_labels`. Given a QA item (with optional
dataset evidence) and the candidate EvidenceGroups recalled for it, produce:

    gold_supporting_groups     all groups whose page is in evidence (or whose
                               text matches the answer)  — WIDE label
    minimal_sufficient_groups  among gold, tiers that can be justified as
                               sufficient — never an unverified cheap fallback

Modes:
    evidence  use dataset-provided evidence pages (+ answer-match fallback)
    distill   use a zero-shot router's selection as the teacher label
    both      evidence when available, else distill
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..data.evidence import (
    has_text_modality,
    has_visual_modality,
    page_modalities,
)
from ..schemas import EvidenceGroup, ModalityBias
from ..utils import dbg


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s;]", " ", (s or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _answer_in_text(answer: str, text: str, mode: str, f1_thr: float) -> bool:
    t = _norm(text)
    aliases = [_norm(part) for part in (answer or "").split(";")]
    aliases = [part for part in aliases if part]
    if not aliases or not t:
        return False
    text_tokens = set(t.split())
    for alias in aliases:
        if alias in t:
            return True
        if mode != "substring":
            answer_tokens = set(alias.split())
            coverage = len(answer_tokens & text_tokens) / max(len(answer_tokens), 1)
            if coverage >= f1_thr:
                return True
    return False


def _qa_answers(qa: dict) -> List[str]:
    aliases = [str(value) for value in (qa.get("answer_aliases") or [])
               if str(value).strip()]
    return aliases or [str(qa.get("answer", ""))]


class AutoLabeler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.labeling.mode
        self.match = cfg.labeling.answer_match
        self.f1_thr = cfg.labeling.token_f1_threshold

    def label(self, qa: dict, groups: List[EvidenceGroup], node_store,
              zeroshot=None, bias: Optional[ModalityBias] = None
              ) -> Optional[Dict[str, List[str]]]:
        """Return {'gold': [...], 'minimal': [...]} or None if unlabelable."""
        gold = self._evidence_gold(qa, groups, node_store)

        if not gold and self.mode in ("distill", "both") and zeroshot is not None:
            sel = zeroshot.rank_and_select(
                list(groups), self.cfg.cost.max_cost_normalizer,
                bias=bias, question=qa.get("question", ""),
            )
            gold = [g.group_id for g in sel]
            if gold:
                dbg(f"[label] {qa['qa_id']} distilled gold={gold}")

        if not gold:
            return None

        minimal = self._minimal_sufficient(qa, groups, gold, node_store)
        return {"gold": gold, "minimal": minimal}

    # ----------------------------------------------------------- evidence gold
    def _evidence_gold(self, qa, groups, node_store) -> List[str]:
        if self.mode == "distill":
            return []
        ev = qa.get("evidence") or {}
        ev_pages = set(ev.get("pages") or [])
        gold_doc = qa.get("qrel_doc_id") or qa.get("doc_id")
        gold = []
        for g in groups:
            root = node_store.get(g.root_node_id)
            if root is None:
                continue
            page = root.source_ref.page_num
            doc_match = not gold_doc or g.source_doc_id == gold_doc
            page_match = doc_match and ((page in ev_pages) if ev_pages else False)
            ans_match = doc_match and any(
                _answer_in_text(answer, root.text, self.match, self.f1_thr)
                for answer in _qa_answers(qa))
            if page_match or ans_match:
                gold.append(g.group_id)
        if gold:
            dbg(f"[label] {qa['qa_id']} evidence gold={gold} (pages={sorted(ev_pages)})")
        return gold

    # ------------------------------------------------------ minimal sufficient
    def _minimal_sufficient(self, qa, groups, gold_ids, node_store) -> List[str]:
        """Return defensible sufficient tiers, one per supporting cluster.

        Text containment is the strongest signal.  When the gold modality says
        visual, a gold screenshot is a defensible fallback.  For textual gold
        evidence, full text is the conservative fallback.  A caption is never
        labeled sufficient merely because it is cheap.
        """
        answers = _qa_answers(qa)
        ev = qa.get("evidence") or {}
        gold_groups = [g for g in groups if g.group_id in set(gold_ids)]
        gold_groups.sort(key=lambda g: g.token_equivalent_cost)

        # group by page (redundancy cluster); within each page pick cheapest sufficient
        by_cluster: Dict[str, List[EvidenceGroup]] = {}
        for g in gold_groups:
            by_cluster.setdefault(g.redundancy_cluster_id, []).append(g)

        minimal = []
        for cluster, gs in by_cluster.items():
            chosen = None
            root_for_modality = next(
                (node_store.get(g.root_node_id) for g in gs
                 if node_store.get(g.root_node_id) is not None), None)
            modalities = page_modalities(
                ev, root_for_modality.source_ref.page_num
                if root_for_modality is not None else None)
            visual_gold = has_visual_modality(modalities)
            textual_gold = has_text_modality(modalities)
            for g in gs:   # already cheapest-first
                root = node_store.get(g.root_node_id)
                if root is None:
                    continue
                if any(_answer_in_text(answer, root.text, self.match, self.f1_thr)
                       for answer in answers):
                    chosen = g.group_id
                    break
            if chosen is None:
                if visual_gold:
                    shot = next((g for g in gs if g.node_class in
                                 ("pdf_page", "image", "pdf_region")), None)
                    chosen = shot.group_id if shot else None
                if chosen is None and textual_gold:
                    full = next((g for g in gs if g.node_class in
                                 ("text_span", "text_cluster")), None)
                    chosen = full.group_id if full else None
            if chosen is not None:
                minimal.append(chosen)
        dbg(f"[label] {qa['qa_id']} minimal={minimal}")
        return minimal
