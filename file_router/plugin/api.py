"""Framework-agnostic entry point for the cost-aware selector.

Deliberately depends on nothing from the host system: a Candidate is a plain
dataclass, and the only required fields are an id, a cost, and a relevance
score.  Everything else (text, image path, tier name) is carried through
untouched so the caller can rebuild its own context after selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..router.selector import build_selector
from ..schemas import EvidenceGroup


# Node classes the selector already understands, keyed by the tier vocabulary
# used throughout the experiments.  An unknown tier is passed through, which
# only affects tier-specific rules (never correctness of selection).
_TIER_TO_NODE_CLASS = {
    "caption": "doc_summary",
    "fulltext": "text_span",
    "screenshot": "pdf_page",
    "image": "image",
    "region": "pdf_region",
}


@dataclass
class Candidate:
    """One retrievable unit, priced.

    id:         caller's identifier; returned untouched so results can be joined.
    cost:       token-equivalent cost of putting this in the prompt.  If the
                caller has no cost model, use `estimate_cost`.
    score:      relevance from the host retriever.  Any monotonic scale works;
                it is normalized internally.
    group_key:  units sharing this key are the SAME evidence at different
                granularities (e.g. a page's caption / text / screenshot).  At
                most one member of a group is selected.  Defaults to `id`, which
                makes every candidate its own group (i.e. no tier choice).
    tier:       optional label used by cost-tier rules.
    payload:    opaque; the caller gets it back on the selected candidates.
    """

    id: str
    cost: float
    score: float
    group_key: Optional[str] = None
    tier: Optional[str] = None
    text: Optional[str] = None
    image_path: Optional[str] = None
    doc_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def resolved_group_key(self) -> str:
        return self.group_key or self.id


@dataclass
class SelectionReport:
    """What was chosen, and what it saved."""

    selected: List[Candidate]
    candidate_count: int
    selected_cost: float
    baseline_cost: float
    tier_counts: Dict[str, int]

    @property
    def cost_saving_rate(self) -> float:
        """Fraction of the baseline prompt cost avoided."""
        if self.baseline_cost <= 0:
            return 0.0
        return (self.baseline_cost - self.selected_cost) / self.baseline_cost

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": [c.id for c in self.selected],
            "candidate_count": self.candidate_count,
            "selected_count": len(self.selected),
            "selected_cost": round(self.selected_cost, 2),
            "baseline_cost": round(self.baseline_cost, 2),
            "cost_saving_rate": round(self.cost_saving_rate, 4),
            "tier_counts": dict(self.tier_counts),
        }


def estimate_cost(text: str = "", *, images: int = 0,
                  chars_per_token: float = 4.0,
                  tokens_per_image: float = 1200.0) -> float:
    """Cost fallback for hosts that do not price their own candidates.

    Mirrors the ingestion-time cost model: text is charged by characters, an
    image by a flat page-screenshot budget.
    """
    return len(text or "") / max(chars_per_token, 1e-6) + images * tokens_per_image


class CostRouter:
    """Cost-aware selection over candidates produced by any retriever."""

    def __init__(self, router_cfg: Any, *, scorer: Any = None,
                 max_cost: Optional[float] = None):
        self.cfg = router_cfg
        self.scorer = scorer
        self.selector = build_selector(router_cfg)
        self.max_cost = (max_cost if max_cost is not None
                         else float(getattr(router_cfg, "max_total_cost", 6000.0)))

    # ---------------------------------------------------------------- factories
    @classmethod
    def from_config(cls, config_path: str = "config/file_router.yaml",
                    *, use_learned_scorer: bool = False,
                    model_path: str = "") -> "CostRouter":
        """Build from the project YAML, optionally with the trained scorer.

        The learned scorer needs features that only exist inside the full
        pipeline (modality bias, node levels), so mounting on a foreign
        retriever defaults to the training-free ranking: the host's own
        relevance score, spent under our cost model.
        """
        from ..config import load_config

        cfg = load_config(config_path)
        scorer = None
        if use_learned_scorer:
            from ..router.scorer import LearnedScorer

            scorer = LearnedScorer(
                cfg.router, cfg.trainer,
                model_path or cfg.paths.router_model_dir)
        return cls(cfg.router, scorer=scorer,
                   max_cost=float(cfg.cost.max_cost_normalizer))

    @classmethod
    def with_defaults(cls, **overrides: Any) -> "CostRouter":
        """Build without the repo config, for use as a standalone dependency.

        `submod_cost_beta` is the knob that decides how hard cheapness is
        chased.  ablation_20260831_032319 measured what each tier is actually
        worth once the Gold page is found: fulltext scored F1 0.3534 at cost
        892, caption 0.1850 at 1625, screenshot 0.3451 at 4978.  A cheap tier
        that cannot answer is not a saving, so the default leans less on raw
        price than the in-repo pipeline, which has a trained scorer to tell the
        tiers apart and does not need the selector to be cautious.
        """
        params: Dict[str, Any] = {
            "strategy": "submod_knapsack",
            "max_groups": 5,
            "max_total_cost": 6000.0,
            "beta": 1.0,
            "confidence_mass_threshold": 0.95,
            "submod_alpha": 1.0,
            "submod_gamma": 0.35,
            "submod_cost_beta": 0.08,
            "min_relative_probability": 0.01,
            "min_distinct_pages": 4,
            "family_mutex": True,
            "expensive_tier_cost": 800.0,
            "expensive_tier_max_rank": 1,
            "expensive_tier_min_probability_share": 0.45,
            "disable_tiers": [],
        }
        params.update(overrides)
        return cls(SimpleNamespace(**params))

    # ------------------------------------------------------------------ select
    def select(self, candidates: Sequence[Candidate]) -> SelectionReport:
        """Choose a cost-efficient subset. Order of the input is irrelevant."""
        candidates = list(candidates)
        if not candidates:
            return SelectionReport([], 0, 0.0, 0.0, {})

        groups = [self._to_group(c) for c in candidates]
        self._assign_probabilities(groups, candidates)
        chosen = self.selector.select(groups)
        if not chosen:
            chosen = [max(groups, key=lambda g: g.router_probability)]

        by_id = {c.id: c for c in candidates}
        selected = [by_id[g.group_id] for g in chosen if g.group_id in by_id]

        tier_counts: Dict[str, int] = {}
        for c in selected:
            tier_counts[c.tier or "unknown"] = tier_counts.get(c.tier or "unknown", 0) + 1

        # Baseline = what the host would have paid by sending its top-N as-is,
        # where N is how many groups we ended up keeping.  Comparing against the
        # whole pool would flatter us by counting candidates nobody would send.
        top_n = sorted(candidates, key=lambda c: -c.score)[:max(len(selected), 1)]
        return SelectionReport(
            selected=selected,
            candidate_count=len(candidates),
            selected_cost=sum(c.cost for c in selected),
            baseline_cost=sum(c.cost for c in top_n),
            tier_counts=tier_counts,
        )

    # ----------------------------------------------------------------- helpers
    def _to_group(self, c: Candidate) -> EvidenceGroup:
        node_class = _TIER_TO_NODE_CLASS.get(
            (c.tier or "").lower(), c.tier or "text_span")
        return EvidenceGroup(
            group_id=c.id,
            root_node_id=c.id,
            node_ids=[c.id],
            node_class=node_class,
            level=0,
            redundancy_cluster_id=c.resolved_group_key(),
            base_score=float(c.score),
            token_equivalent_cost=float(c.cost),
            coverage_signature=None,
            source_doc_id=c.doc_id,
        )

    def _assign_probabilities(self, groups: List[EvidenceGroup],
                              candidates: List[Candidate]) -> None:
        """Turn host relevance scores into a probability distribution.

        Softmax over min-max normalized scores.  Host scores arrive on wildly
        different scales (cosine similarity, BM25, inverse distance), so the
        normalization is what makes the selector's relative thresholds behave
        the same regardless of which system it is mounted on.
        """
        if self.scorer is not None:
            self.scorer.score(groups, self.max_cost)
            return

        scores = [float(c.score) for c in candidates]
        lo, hi = min(scores), max(scores)
        spread = hi - lo
        if spread <= 1e-9:
            uniform = 1.0 / len(groups)
            for g in groups:
                g.router_probability = uniform
                g.router_score = 0.0
            return

        # Temperature 4 keeps the distribution peaked enough that the relative
        # floors discriminate, without collapsing onto a single candidate.
        weights = [math.exp(4.0 * (s - lo) / spread) for s in scores]
        total = sum(weights)
        for g, w, s in zip(groups, weights, scores):
            g.router_probability = w / total
            g.router_score = s
