"""The BM25 and prompting baselines added for the main comparison table.

Both are meant to differ from Vanilla RAG in exactly one respect, so the tests
pin that: BM25 changes which pages are ranked first, and never changes the
granularity rule or the budget.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from file_router.schemas import EvidenceGroup  # noqa: E402
import evaluate_validation as ev  # noqa: E402


class _Nodes:
    """Minimal node store: the baseline only needs id -> text."""

    def __init__(self, texts):
        self._texts = texts

    def get(self, node_id):
        text = self._texts.get(node_id)
        return SimpleNamespace(text=text) if text is not None else None


def _group(name, cluster, text, cost, node_class="text_span", score=0.5):
    return EvidenceGroup(
        group_id=name, root_node_id=name, node_ids=[name],
        node_class=node_class, level=0, redundancy_cluster_id=cluster,
        router_probability=score, token_equivalent_cost=cost,
        coverage_signature=cluster, source_doc_id="doc",
        base_score=score,
    )


def _cfg():
    return SimpleNamespace(router=SimpleNamespace(
        max_total_cost=6000.0, max_groups=5))


def test_bm25_ranks_the_lexically_matching_page_first():
    """Dense order puts the wrong page first; BM25 must override it."""
    groups = [
        _group("a", "p1", "unrelated boilerplate text", 100, score=0.9),
        _group("b", "p2", "quarterly revenue grew twelve percent", 100,
               score=0.1),
    ]
    nodes = _Nodes({"a": "unrelated boilerplate text",
                    "b": "quarterly revenue grew twelve percent"})
    picked = ev._retrieval_baseline_groups(
        groups, _cfg(), "what was quarterly revenue growth", "bm25", nodes)
    assert picked[0].redundancy_cluster_id == "p2"

    # The dense arm sees the same candidates and keeps the base-score order,
    # which is what makes the two arms a controlled comparison.
    picked_dense = ev._retrieval_baseline_groups(
        groups, _cfg(), "what was quarterly revenue growth", "dense", nodes)
    assert picked_dense[0].redundancy_cluster_id == "p1"


def test_retrieval_baselines_respect_the_shared_budget():
    groups = [_group(f"g{i}", f"p{i}", "text", 2000.0) for i in range(6)]
    picked = ev._retrieval_baseline_groups(
        groups, _cfg(), "q", "dense", _Nodes({}))
    assert len(picked) <= 5
    assert sum(g.token_equivalent_cost for g in picked) <= 6000.0


def test_retrieval_baselines_match_vanilla_rag_given_the_same_order():
    """The arms must differ from Vanilla RAG in ranking only.

    Given an order identical to dense recall, the dense arm has to reproduce
    Vanilla RAG's selection exactly, including its budget behaviour: it spends
    on the richest affordable tier even when doing so exhausts the budget on
    one page. Any divergence here would make the comparison confound ranking
    with granularity policy.
    """
    groups = [
        _group("rich", "p1", "revenue", 5000.0, score=0.9),
        _group("cheap", "p1", "revenue", 100.0,
               node_class="doc_summary", score=0.9),
        _group("other", "p2", "revenue", 4000.0, score=0.5),
    ]
    nodes = _Nodes({"rich": "revenue", "cheap": "revenue", "other": "revenue"})
    picked = ev._retrieval_baseline_groups(
        groups, _cfg(), "revenue", "dense", nodes)
    vanilla = ev._full_context_groups(groups, _cfg())
    assert [g.group_id for g in picked] == [g.group_id for g in vanilla]
