"""Zero-shot cost-ladder router (cold-start / distillation teacher).

A minimal, training-free scorer that bakes in the core preference
"caption < screenshot < full-text" directly:

    score(g) = base_score(g) - cost_penalty * normalized_cost(g)
             + tier_prior(g)            # caption gets a bonus, full-text a small malus

Then family_mutex (one tier per page) + a small budget keeps the cheapest
tier whose relevance is high enough. Used by:
  - evaluate.py  ("zeroshot" router column), and
  - labeling.py  (distill mode: its selection becomes a teacher label).

It reuses the same Selector machinery so its output is directly comparable
to the rule-based and learned routers.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..schemas import EvidenceGroup, ModalityBias
from .selector import build_selector


# tier prior keyed by node_class (caption=doc_summary is cheapest & preferred)
_TIER_PRIOR = {
    "doc_summary": 0.15,   # caption — nudge up
    "pdf_page": 0.0,       # screenshot — neutral
    "text_span": -0.05,    # full text — slight malus (only win if clearly needed)
    "text_cluster": 0.0,
    "image": 0.0,
    "pdf_region": 0.0,
}


class ZeroShotScorer:
    def __init__(self, cfg_router, cost_penalty: float = 0.5):
        self.cfg = cfg_router
        self.cost_penalty = cost_penalty

    def score(self, groups: List[EvidenceGroup], max_cost: float,
              bias: ModalityBias | None = None, question: str = ""):
        if not groups:
            return
        raw = []
        for g in groups:
            norm_cost = g.token_equivalent_cost / max(max_cost, 1.0)
            s = (g.base_score
                 + _TIER_PRIOR.get(g.node_class, 0.0)
                 - self.cost_penalty * norm_cost)
            g.router_score = s
            raw.append(s)
        raw = np.asarray(raw, dtype=np.float64)
        raw = raw - raw.max()
        probs = np.exp(raw)
        probs = probs / (probs.sum() + 1e-9)
        for g, p in zip(groups, probs):
            g.router_probability = float(p)


class ZeroShotRouter:
    """Plug-compatible with RouterPipeline's scorer+selector pair."""

    def __init__(self, cfg):
        self.scorer = ZeroShotScorer(cfg.router)
        self.selector = build_selector(cfg.router)

    def rank_and_select(self, groups: List[EvidenceGroup], max_cost: float,
                        bias: ModalityBias | None = None, question: str = ""):
        self.scorer.score(groups, max_cost, bias=bias, question=question)
        selected = self.selector.select(groups)
        if not selected and groups:
            selected = sorted(groups, key=lambda g: -g.router_probability)[:1]
        return selected
