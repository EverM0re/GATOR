"""Mount File_Router's cost-aware selector on top of someone else's retriever.

The claim this package exists to test: choosing the *granularity* at which
already-retrieved evidence is read is orthogonal to choosing *which* evidence to
retrieve.  If that holds, any RAG or memory system can keep its own retriever and
still cut token cost by routing granularity -- no retraining, no re-indexing.

Usage is two calls:

    from file_router.plugin import CostRouter, Candidate

    router = CostRouter.from_config("config/file_router.yaml")
    kept = router.select([
        Candidate(id="doc1_p3_full", text=page_text, cost=340, score=0.81,
                  group_key="doc1:p3", tier="fulltext"),
        Candidate(id="doc1_p3_cap", text=caption, cost=48, score=0.77,
                  group_key="doc1:p3", tier="caption"),
        ...
    ])

`group_key` is what makes this work: candidates sharing one carry the same
underlying evidence at different prices, so at most one is bought.
"""

from .api import Candidate, CostRouter, SelectionReport

__all__ = ["Candidate", "CostRouter", "SelectionReport"]
