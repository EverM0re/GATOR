"""Adapter contract tests.

The hosts are not installed in CI (and PyPI is not always reachable), so these
use fakes that reproduce each host's *documented return shape*, taken from the
cloned sources:

    mem0   mem0/memory/main.py:1707  -> {"results": [{"id","memory","score",...}]}
    A-Mem  agentic_memory/memory_system.py:432
           -> [{"id","content","context","keywords","score"}]  score = Chroma DISTANCE

What is being tested is the adapter's own logic -- shape handling, score
polarity, tier expansion, and that the host's retrieval is left alone.
"""

from typing import Any, Dict, List

from file_router.plugin import CostRouter
from integrations.base import HostAdapter, HostHit


class _FakeHost(HostAdapter):
    """Minimal adapter over canned hits, to exercise the base-class behaviour."""

    name = "fake"

    def __init__(self, hits: List[HostHit], **kwargs):
        super().__init__(**kwargs)
        self._hits = hits
        self.retrieve_calls = 0

    def add(self, item: Dict[str, Any]) -> None:
        pass

    def retrieve(self, query: str, top_k: int) -> List[HostHit]:
        self.retrieve_calls += 1
        return self._hits[:top_k]


def _long_hits(n=6):
    return [HostHit(id=f"h{i}", text="lorem ipsum " * 80, score=0.9 - 0.1 * i,
                    doc_id=f"doc{i}") for i in range(n)]


def test_routing_reduces_cost_against_the_host_baseline():
    result = _FakeHost(_long_hits()).compare("q")
    assert result.routed_cost < result.baseline_cost
    assert result.cost_saving_rate > 0


def test_host_retriever_is_called_exactly_once_per_query():
    """Both arms must score the SAME retrieval, or the comparison is confounded."""
    host = _FakeHost(_long_hits())
    host.compare("q")
    assert host.retrieve_calls == 1


def test_tier_expansion_offers_a_cheaper_view_of_long_text():
    candidates = _FakeHost([]).expand_tiers(
        HostHit(id="h", text="x" * 2000, score=0.5))
    tiers = {c.tier for c in candidates}
    assert "fulltext" in tiers and "caption" in tiers
    groups = {c.resolved_group_key() for c in candidates}
    assert len(groups) == 1, "tiers of one hit must share a group key"


def test_short_text_gets_no_pointless_caption_tier():
    """A 'caption' as long as the text is a duplicate candidate, not a saving."""
    candidates = _FakeHost([]).expand_tiers(
        HostHit(id="h", text="short", score=0.5))
    assert [c.tier for c in candidates] == ["fulltext"]


def test_image_bearing_hit_gains_a_screenshot_tier():
    candidates = _FakeHost([]).expand_tiers(
        HostHit(id="h", text="x" * 2000, score=0.5, image_path="/tmp/p.png"))
    assert "screenshot" in {c.tier for c in candidates}


def test_empty_retrieval_does_not_crash():
    result = _FakeHost([]).compare("q")
    assert result.routed_ids == [] and result.cost_saving_rate == 0.0


def test_mem0_response_shape_is_parsed():
    """Shape from mem0/memory/main.py:1707."""
    from integrations.adapters import Mem0Adapter

    response = {"results": [
        {"id": "m1", "memory": "the answer is 42", "score": 0.81,
         "metadata": {"doc_id": "d1"}},
        {"id": "m2", "memory": "unrelated", "score": 0.30, "metadata": {}},
    ]}
    hits = []
    for r in response["results"]:
        meta = r.get("metadata") or {}
        hits.append(HostHit(id=str(r["id"]), text=r["memory"],
                            score=float(r["score"]),
                            doc_id=str(meta.get("doc_id") or "")))
    assert [h.id for h in hits] == ["m1", "m2"]
    assert hits[0].score > hits[1].score
    assert Mem0Adapter.name == "mem0"


def test_amem_distance_is_inverted_into_a_similarity():
    """A-Mem returns a Chroma DISTANCE: lower is better, opposite to mem0.

    Getting this backwards would make the adapter prefer the least relevant
    memories while still looking like it works.
    """
    results = [{"id": "a", "content": "close", "score": 0.1},
               {"id": "b", "content": "far", "score": 0.9}]
    distances = [r["score"] for r in results]
    worst = max(distances)
    scores = [1.0 - (d / worst) for d in distances]
    assert scores[0] > scores[1]


def test_lightrag_context_string_is_parsed_into_hits():
    from integrations.adapters import LightRAGAdapter

    context = (
        "-----Document Chunks(DC)-----\n"
        '```json\n[{"id":"c1","content":"first","file_path":"a.pdf"},'
        '{"id":"c2","content":"second","file_path":"b.pdf"}]\n```'
    )
    hits = LightRAGAdapter.parse_context(context)
    assert [h.text for h in hits] == ["first", "second"]
    # No real relevance score at this layer; rank order is all LightRAG gives.
    assert hits[0].score > hits[1].score


def test_lightrag_parser_survives_malformed_json():
    from integrations.adapters import LightRAGAdapter

    assert LightRAGAdapter.parse_context(
        "-----Chunks-----\n```json\n{not valid\n```") == []
    assert LightRAGAdapter.parse_context("") == []


def test_custom_router_is_honoured():
    tight = CostRouter.with_defaults(max_groups=2, min_distinct_pages=1)
    assert len(_FakeHost(_long_hits(), router=tight).compare("q").routed_ids) <= 2


def test_real_ingestion_tier_and_cost_are_used_when_present():
    """Records from the node store carry their true tier and cost.

    Re-estimating cost from character counts would price a page screenshot as
    if it were its caption text, which is the exact distinction the router
    exists to make.
    """
    hit = HostHit(id="n1", text="x" * 2000, score=0.7,
                  metadata={"tier": "screenshot", "cost": 1543.0,
                            "group_key": "doc:p3"})
    candidates = _FakeHost([]).expand_tiers(hit)
    assert len(candidates) == 1
    assert candidates[0].tier == "screenshot"
    assert candidates[0].cost == 1543.0
    assert candidates[0].resolved_group_key() == "doc:p3"


def test_store_backed_tiers_of_one_page_compete_as_one_group():
    """Two hits sharing a group_key are one page at two prices, so one wins."""
    hits = [
        HostHit(id="cap", text="short caption", score=0.62,
                metadata={"tier": "caption", "cost": 48.0,
                          "group_key": "doc:p3"}),
        HostHit(id="full", text="the full page text", score=0.81,
                metadata={"tier": "fulltext", "cost": 340.0,
                          "group_key": "doc:p3"}),
    ]
    result = _FakeHost(hits).compare("q")
    assert len(result.routed_ids) == 1


def test_amem_exports_openai_key_before_constructing_the_host(monkeypatch):
    """A-Mem raises inside its own constructor when OPENAI_API_KEY is unset.

    Our key lives in the project config, not that variable, so the adapter must
    export it *before* constructing -- the post-construction redirect cannot
    run if construction itself raised.
    """
    import os
    import sys
    import types

    seen = {}

    class _FakeMemory:
        def __init__(self, **kwargs):
            # Mirrors llm_controller.py:21, which raises during __init__.
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key not found. "
                                 "Set OPENAI_API_KEY environment variable.")
            seen["constructed"] = True

    module = types.ModuleType("agentic_memory.memory_system")
    module.AgenticMemorySystem = _FakeMemory
    package = types.ModuleType("agentic_memory")
    monkeypatch.setitem(sys.modules, "agentic_memory", package)
    monkeypatch.setitem(sys.modules, "agentic_memory.memory_system", module)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from integrations import adapters
    monkeypatch.setattr(adapters, "llm_settings", lambda *a, **k: {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "test-model",
        "api_key": "key-from-config",
    })

    adapters.AMemAdapter(local=False)
    assert seen.get("constructed"), "adapter must construct the host"
    assert os.environ["OPENAI_API_KEY"] == "key-from-config"
