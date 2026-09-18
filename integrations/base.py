"""Shared contract for host adapters.

A host adapter is deliberately thin.  It must not improve the host's retrieval:
the experiment compares the host's own top-k against the same top-k re-priced by
CostRouter, so any retrieval change would confound the result.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from file_router.plugin import Candidate, CostRouter


@dataclass
class HostHit:
    """One result from the host's retriever, normalized across systems."""

    id: str
    text: str
    score: float
    doc_id: str = ""
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Host-alone vs host+router, for one query."""

    query: str
    host: str
    baseline_ids: List[str]
    baseline_cost: float
    baseline_text: str
    routed_ids: List[str]
    routed_cost: float
    routed_text: str
    candidate_count: int
    tier_counts: Dict[str, int]
    retrieval_ms: float
    routing_ms: float

    @property
    def cost_saving_rate(self) -> float:
        if self.baseline_cost <= 0:
            return 0.0
        return (self.baseline_cost - self.routed_cost) / self.baseline_cost

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "host": self.host,
            "baseline_ids": self.baseline_ids,
            "baseline_cost": round(self.baseline_cost, 2),
            "routed_ids": self.routed_ids,
            "routed_cost": round(self.routed_cost, 2),
            "cost_saving_rate": round(self.cost_saving_rate, 4),
            "candidate_count": self.candidate_count,
            "kept_count": len(self.routed_ids),
            "tier_counts": self.tier_counts,
            "retrieval_ms": round(self.retrieval_ms, 2),
            "routing_ms": round(self.routing_ms, 2),
        }


class HostAdapter(ABC):
    """Wraps one memory system so its retrieval can be re-priced."""

    name: str = "host"

    def __init__(self, router: Optional[CostRouter] = None,
                 *, top_k: int = 10):
        self.router = router or CostRouter.with_defaults()
        self.top_k = top_k

    # ------------------------------------------------------ host-specific
    @abstractmethod
    def add(self, item: Dict[str, Any]) -> None:
        """Ingest one unified-format record into the host."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[HostHit]:
        """Call the HOST's retriever. Must not be modified by the adapter."""

    def expand_tiers(self, hit: HostHit) -> List[Candidate]:
        """Offer the same evidence at several prices.

        When the record came from the node store it already carries its real
        ingestion tier and cost, and several hits can share one `group_key` --
        those are genuine granularities of one page, so no synthetic tier is
        invented.  Otherwise fall back to a truncated view of the text, which is
        the only cheaper rung available for a plain text host.
        """
        from file_router.plugin.api import estimate_cost

        full = hit.text or ""
        group = hit.metadata.get("group_key") or hit.doc_id or hit.id

        real_tier = hit.metadata.get("tier")
        real_cost = hit.metadata.get("cost")
        if real_tier and isinstance(real_cost, (int, float)):
            return [Candidate(
                id=f"{hit.id}::{real_tier}", cost=float(real_cost),
                score=hit.score, group_key=group, tier=real_tier, text=full,
                image_path=hit.image_path, doc_id=hit.doc_id,
                payload={"host_id": hit.id})]

        out = [Candidate(
            id=f"{hit.id}::fulltext", cost=estimate_cost(full), score=hit.score,
            group_key=group, tier="fulltext", text=full, doc_id=hit.doc_id,
            payload={"host_id": hit.id})]

        head = full[:240]
        if len(full) > 480:
            # Only a real saving when the truncation is substantial; a 'caption'
            # the same size as the text would just add a duplicate candidate.
            out.append(Candidate(
                id=f"{hit.id}::caption", cost=estimate_cost(head),
                score=hit.score * 0.8, group_key=group, tier="caption",
                text=head, doc_id=hit.doc_id, payload={"host_id": hit.id}))
        if hit.image_path:
            out.append(Candidate(
                id=f"{hit.id}::screenshot", cost=estimate_cost("", images=1),
                score=hit.score * 0.9, group_key=group, tier="screenshot",
                image_path=hit.image_path, doc_id=hit.doc_id,
                payload={"host_id": hit.id}))
        return out

    # ---------------------------------------------------------- comparison
    def compare(self, query: str) -> ComparisonResult:
        """Run the host alone, then the host + router, on identical retrieval."""
        from file_router.plugin.api import estimate_cost

        started = time.perf_counter()
        hits = self.retrieve(query, self.top_k)
        retrieval_ms = (time.perf_counter() - started) * 1000.0

        # Control: what the host would send today -- every hit, full text.
        baseline_cost = sum(estimate_cost(h.text or "") +
                            (estimate_cost("", images=1) if h.image_path else 0.0)
                            for h in hits)
        baseline_text = "\n\n".join(h.text or "" for h in hits)

        candidates: List[Candidate] = []
        for hit in hits:
            candidates.extend(self.expand_tiers(hit))

        started = time.perf_counter()
        report = self.router.select(candidates)
        routing_ms = (time.perf_counter() - started) * 1000.0

        return ComparisonResult(
            query=query, host=self.name,
            baseline_ids=[h.id for h in hits],
            baseline_cost=baseline_cost,
            baseline_text=baseline_text,
            routed_ids=[c.id for c in report.selected],
            routed_cost=report.selected_cost,
            routed_text="\n\n".join(c.text or "" for c in report.selected),
            candidate_count=len(candidates),
            tier_counts=report.tier_counts,
            retrieval_ms=retrieval_ms, routing_ms=routing_ms,
        )
