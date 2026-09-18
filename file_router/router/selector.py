"""Cost-sensitive selectors.

GreedySelector        — top-M by p_i / cost_i^beta, with family-mutex.
SubmodKnapsackSelector — greedy maximization of
                           α·Σp_i − γ·Σ_{i,j∈S} sim(i,j)·[same cluster]
                         under cost budget + family-mutex + |S|≤M.

Both return a list of selected EvidenceGroup, preserving order.

Both can also stop on a relative-probability floor. This prevents a peaked
learned policy from appending very low-confidence evidence merely because it
is cheap or embedding-orthogonal to the first choice.

The floor is applied *relative to the running best remaining candidate*, not to
the very first pick.  Anchoring it to the first pick made the threshold behave
like an absolute cutoff: with ~80 candidates the learned softmax is flat (top
probability ~0.12), so a 5% relative floor pruned nearly every runner-up and
collapsed the selection to <2 groups while >80% of the cost budget went unused.

Both selectors also honour `min_distinct_pages`: until that many distinct
source pages are in the set, the floor and the redundancy penalty are not
allowed to terminate selection.  Gold-page recall, not tier choice, is what
bounds end-to-end answer quality, so spending idle budget on page diversity is
strictly better than returning early.

`expensive_tier_max_rank` then keeps that page quota from being paid for in the
most expensive currency.  In campaign_20260824_035133 the quota worked — Gold
page hit rate went 0.31 -> 0.50 — but 95% of screenshot spend (110 of 116
picks) landed on non-Gold pages, because a screenshot was never the first pick
and only ever appeared at ranks 3-4 as quota filler at ~1550 tokens each.  A
page worth 13x a caption has to be worth it on relevance, not on being the
last slot left, so expensive groups are only eligible while the selector is
still making high-confidence picks.
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np

from ..schemas import EvidenceGroup


def _sig_matrix(groups: List[EvidenceGroup]) -> Optional[np.ndarray]:
    sigs = []
    dim = None
    for g in groups:
        if g.coverage_signature:
            sigs.append(np.asarray(g.coverage_signature, dtype=np.float32))
            dim = len(sigs[-1])
        else:
            sigs.append(None)
    if dim is None:
        return None
    M = np.zeros((len(groups), dim), dtype=np.float32)
    for i, s in enumerate(sigs):
        if s is not None and len(s) == dim:
            M[i] = s / (np.linalg.norm(s) + 1e-9)
    return M


def _page_key(g: EvidenceGroup) -> str:
    """Identity of the underlying source page, independent of cost tier.

    `redundancy_cluster_id` already encodes doc+page for ingested nodes; fall
    back to the doc id so a missing cluster never collapses every candidate
    into one bucket.
    """
    return g.redundancy_cluster_id or g.source_doc_id or g.group_id


def _min_distinct_pages(cfg) -> int:
    return max(0, int(getattr(cfg, "min_distinct_pages", 0)))


# Cost tier per node class, mirroring TIER_NAMES in scripts/evaluate_validation.
# Kept here so a tier ablation applies inside the selector itself and therefore
# to every router that shares it, rather than only to the reporting layer.
_TIER_OF_CLASS = {
    "doc_summary": "caption",
    "pdf_page": "screenshot",
    "text_span": "fulltext",
}


def _tier_name(g: EvidenceGroup) -> str:
    return _TIER_OF_CLASS.get(g.node_class, g.node_class)


def _disabled_tiers(cfg) -> Set[str]:
    """Tiers this run is forbidden to select, for ablation experiments.

    Accepts a list or a comma-separated string so it can come from YAML or from
    an environment variable unchanged.
    """
    raw = getattr(cfg, "disable_tiers", None)
    if not raw:
        return set()
    if isinstance(raw, str):
        raw = raw.split(",")
    return {str(item).strip().lower() for item in raw if str(item).strip()}


def _expensive_rank_limit(cfg) -> int:
    """Rank after which an expensive group may no longer be selected.

    Negative disables the rule.  Rank is 0-based over the groups already
    chosen, so 1 means "only the first two picks may be expensive".
    """
    return int(getattr(cfg, "expensive_tier_max_rank", -1))


def _is_expensive(g: EvidenceGroup, cfg) -> bool:
    """Expensive relative to the run's own cost scale, not a hard-coded tier.

    Keyed off cost rather than node_class so the rule keeps working if the cost
    model or the tier names change.
    """
    threshold = float(getattr(cfg, "expensive_tier_cost", 0.0))
    if threshold <= 0:
        return False
    return float(g.token_equivalent_cost) >= threshold


def _expensive_blocked(g: EvidenceGroup, cfg, selected_count: int,
                       best_probability) -> bool:
    """Block an expensive group that is being added as low-confidence filler.

    Two independent ways to earn an expensive slot, so a genuinely visual
    question can still buy its page image:
      * rank — it is one of the first `expensive_tier_max_rank + 1` picks, or
      * confidence — its probability is a large fraction of the best pick's.

    Without the confidence escape the rule would permanently cap visual-Gold
    questions, which need the screenshot to be answerable at all.
    """
    limit = _expensive_rank_limit(cfg)
    if limit < 0 or not _is_expensive(g, cfg):
        return False
    if selected_count <= limit:
        return False
    # Deliberately relative, never absolute.  Every router shares this selector,
    # and their softmax scales differ by an order of magnitude: in
    # campaign_20260829_013103 the learned scorer's probabilities had median
    # 0.045 while zeroshot's never exceeded 0.019.  An absolute floor of 0.05
    # therefore blocked 100% of zeroshot's expensive picks and only 57% of the
    # learned router's -- it made the *baseline* cheaper than the router and
    # inverted cost_saving_vs_zeroshot to -41.6% in campaign_20260830_123023.
    # Any threshold added here must be expressed relative to the same router's
    # own distribution.
    share = float(getattr(cfg, "expensive_tier_min_probability_share", 0.0))
    if share > 0 and best_probability:
        if g.router_probability >= best_probability * share:
            return False
    return True


def _apply_tier_ablation(groups: List[EvidenceGroup], cfg
                         ) -> List[EvidenceGroup]:
    """Drop disabled tiers, but never hand the selector an empty pool.

    If every candidate for a query belongs to a disabled tier, the ablation is
    skipped for that query: returning nothing would silently turn the run into
    a no-evidence baseline and confound the comparison the ablation exists to
    make.
    """
    disabled = _disabled_tiers(cfg)
    if not disabled:
        return groups
    kept = [g for g in groups if _tier_name(g) not in disabled]
    return kept or groups


class GreedySelector:
    def __init__(self, cfg):
        self.cfg = cfg

    def select(self, groups: List[EvidenceGroup]) -> List[EvidenceGroup]:
        if not groups:
            return []
        groups = _apply_tier_ablation(groups, self.cfg)
        eps = 1e-6
        scored = sorted(
            groups,
            key=lambda g: -(g.router_probability /
                            ((g.token_equivalent_cost + eps) ** self.cfg.beta)),
        )
        used_clusters: Set[str] = set()
        pages: Set[str] = set()
        selected: List[EvidenceGroup] = []
        mass, cost = 0.0, 0.0
        best_probability = None
        relative_floor = max(
            0.0, float(getattr(self.cfg, "min_relative_probability", 0.05)))
        page_quota = _min_distinct_pages(self.cfg)
        for g in scored:
            if self.cfg.family_mutex and g.redundancy_cluster_id in used_clusters:
                continue
            if cost + g.token_equivalent_cost > self.cfg.max_total_cost:
                continue
            # An expensive tier bought late is quota filler, not evidence.
            if _expensive_blocked(g, self.cfg, len(selected), best_probability):
                continue
            page = _page_key(g)
            # While the page quota is unmet, a new page is always worth its
            # budget: the floor and the mass stop only guard against padding an
            # already page-diverse set.
            needs_page = page not in pages and len(pages) < page_quota
            if (not needs_page and best_probability is not None and
                    relative_floor > 0 and
                    g.router_probability < best_probability * relative_floor):
                continue
            selected.append(g)
            if best_probability is None:
                best_probability = g.router_probability
            used_clusters.add(g.redundancy_cluster_id)
            pages.add(page)
            mass += g.router_probability
            cost += g.token_equivalent_cost
            if len(selected) >= self.cfg.max_groups:
                break
            if mass >= self.cfg.confidence_mass_threshold and len(pages) >= page_quota:
                break
        return selected


class SubmodKnapsackSelector:
    """Lazy greedy for α·relevance − γ·redundancy under cost budget."""

    def __init__(self, cfg):
        self.cfg = cfg

    def select(self, groups: List[EvidenceGroup]) -> List[EvidenceGroup]:
        if not groups:
            return []
        groups = _apply_tier_ablation(groups, self.cfg)
        sig_M = _sig_matrix(groups)            # (N, D) or None
        selected: List[int] = []
        used_clusters: Set[str] = set()
        pages: Set[str] = set()
        cost = 0.0
        candidates = list(range(len(groups)))
        best_probability = None
        relative_floor = max(
            0.0, float(getattr(self.cfg, "min_relative_probability", 0.05)))
        page_quota = _min_distinct_pages(self.cfg)

        def marginal(cand_idx: int) -> float:
            g = groups[cand_idx]
            rel = self.cfg.submod_alpha * g.router_probability
            if not selected or sig_M is None:
                return rel
            sim = sig_M[cand_idx] @ sig_M[selected].T
            # facility-location style: penalize how much this overlaps with current set
            penalty = self.cfg.submod_gamma * float(np.clip(sim, 0.0, 1.0).mean())
            # An unseen page cannot be redundant with the current set in the way
            # that matters here (answer coverage), and embedding similarity
            # between two pages of the same report is high enough to suppress
            # genuinely new evidence.  Only damp the penalty, never invert it.
            if _page_key(g) not in pages:
                penalty *= 0.25
            return rel - penalty

        def cost_adjusted_gain(cand_idx: int) -> float:
            gain = marginal(cand_idx)
            if gain <= 0:
                return gain
            beta = float(getattr(self.cfg, "submod_cost_beta", 0.0))
            if beta <= 0:
                return gain
            normalized_cost = max(
                groups[cand_idx].token_equivalent_cost /
                max(float(self.cfg.max_total_cost), 1.0),
                1e-3,
            )
            return gain / (normalized_cost ** beta)

        while candidates and len(selected) < self.cfg.max_groups:
            need_pages = len(pages) < page_quota
            best_i, best_gain = None, -1e9
            for i in candidates:
                g = groups[i]
                if self.cfg.family_mutex and g.redundancy_cluster_id in used_clusters:
                    continue
                if cost + g.token_equivalent_cost > self.cfg.max_total_cost:
                    continue
                # Applies even while the page quota is unmet: covering one more
                # page is not worth 13x the tokens when a cheaper tier of some
                # other uncovered page is still available.
                if _expensive_blocked(g, self.cfg, len(selected),
                                      best_probability):
                    continue
                new_page = _page_key(g) not in pages
                # Same rule as GreedySelector: the floor may not block a page we
                # have not covered yet while the quota is unmet.
                if not (need_pages and new_page):
                    if (best_probability is not None and relative_floor > 0 and
                            g.router_probability <
                            best_probability * relative_floor):
                        continue
                m = cost_adjusted_gain(i)
                if need_pages and not new_page:
                    # Rank pages we still lack ahead of extra tiers of pages we
                    # already hold, without discarding the latter outright.
                    m -= 1e-6
                if m > best_gain:
                    best_gain = m
                    best_i = i
            if best_i is None:
                break
            # A non-positive gain is only a stop signal once the set is already
            # page-diverse; otherwise the redundancy penalty would end selection
            # with most of the budget unspent.
            if best_gain <= 0 and not (need_pages and
                                       _page_key(groups[best_i]) not in pages):
                break
            selected.append(best_i)
            if best_probability is None:
                best_probability = groups[best_i].router_probability
            used_clusters.add(groups[best_i].redundancy_cluster_id)
            pages.add(_page_key(groups[best_i]))
            cost += groups[best_i].token_equivalent_cost
            candidates.remove(best_i)

        return [groups[i] for i in selected]


def build_selector(cfg):
    if cfg.strategy == "submod_knapsack":
        return SubmodKnapsackSelector(cfg)
    return GreedySelector(cfg)
