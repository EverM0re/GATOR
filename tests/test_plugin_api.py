"""The plugin is the artifact other systems mount, so its contract is tested
independently of the in-repo pipeline."""

from file_router.plugin import Candidate, CostRouter
from file_router.plugin.api import estimate_cost


def _tiered_candidates(pages=6, caption_wins=False):
    """Three granularities per page, priced as measured in the experiments.

    caption_wins flips the relevance order so the cheap tier also looks most
    relevant -- the case where following the host's score and following cost
    point the same way.
    """
    out = []
    for page in range(pages):
        base = 0.90 - 0.1 * page
        cap_score = base if caption_wins else base * 0.80
        out += [
            Candidate(id=f"p{page}_cap", cost=48.0, score=cap_score,
                      group_key=f"doc:p{page}", tier="caption"),
            Candidate(id=f"p{page}_full", cost=340.0, score=base,
                      group_key=f"doc:p{page}", tier="fulltext"),
            Candidate(id=f"p{page}_shot", cost=1500.0, score=base * 0.90,
                      group_key=f"doc:p{page}", tier="screenshot"),
        ]
    return out


def test_at_most_one_granularity_per_group():
    """The whole premise: same evidence at three prices, buy one."""
    report = CostRouter.with_defaults().select(_tiered_candidates())
    groups = [c.resolved_group_key() for c in report.selected]
    assert len(groups) == len(set(groups))


def test_selection_is_cheaper_than_sending_the_top_n_verbatim():
    report = CostRouter.with_defaults().select(_tiered_candidates())
    assert report.selected_cost < report.baseline_cost
    assert report.cost_saving_rate > 0


def test_follows_host_relevance_rather_than_only_price():
    """A cheap tier that the host ranked lower must not automatically win.

    ablation_20260831_032319: caption was the cheapest tier and the worst one
    (F1 0.185 vs fulltext 0.353), so a selector that always buys the cheapest
    rung reproduces the failure that experiment diagnosed.
    """
    report = CostRouter.with_defaults().select(_tiered_candidates())
    assert report.tier_counts.get("fulltext", 0) > 0
    assert report.tier_counts.get("caption", 0) == 0


def test_cheapest_tier_is_used_when_the_host_ranks_it_first():
    report = CostRouter.with_defaults().select(
        _tiered_candidates(caption_wins=True))
    assert report.tier_counts.get("caption", 0) > 0


def test_empty_input_is_not_an_error():
    report = CostRouter.with_defaults().select([])
    assert report.selected == []
    assert report.cost_saving_rate == 0.0


def test_never_returns_empty_when_candidates_exist():
    """Hosts substitute our output for their retriever; returning nothing would
    silently turn their pipeline into a no-context baseline."""
    single = [Candidate(id="only", cost=99999.0, score=0.5)]
    assert CostRouter.with_defaults().select(single).selected


def test_candidates_without_group_key_are_independent():
    """No group_key means no granularity choice -- plain cost-aware top-k."""
    flat = [Candidate(id=f"c{i}", cost=100.0, score=1.0 - 0.1 * i)
            for i in range(8)]
    report = CostRouter.with_defaults().select(flat)
    assert len({c.id for c in report.selected}) == len(report.selected)


def test_payload_and_text_survive_selection():
    """The host must be able to rebuild its own prompt from what we return."""
    marked = [Candidate(id="a", cost=10.0, score=0.9, text="hello",
                        payload={"src": "host"})]
    selected = CostRouter.with_defaults().select(marked).selected[0]
    assert selected.text == "hello"
    assert selected.payload["src"] == "host"


def test_score_scale_does_not_change_the_selection():
    """Hosts return cosine, BM25, or inverse distance; the result must not
    depend on which."""
    base = _tiered_candidates()
    scaled = [Candidate(id=c.id, cost=c.cost, score=c.score * 100.0 + 50.0,
                        group_key=c.group_key, tier=c.tier) for c in base]
    a = [c.id for c in CostRouter.with_defaults().select(base).selected]
    b = [c.id for c in CostRouter.with_defaults().select(scaled).selected]
    assert a == b


def test_estimate_cost_prices_text_and_images():
    assert estimate_cost("x" * 400) == 100.0
    assert estimate_cost("", images=2) == 2400.0
