from pathlib import Path
from types import SimpleNamespace

from file_router.router.selector import SubmodKnapsackSelector
from file_router.schemas import EvidenceGroup


def _group(name, probability, cost, signature):
    return EvidenceGroup(
        group_id=name, root_node_id=name, node_ids=[name],
        node_class="text_span", level=0, redundancy_cluster_id=name,
        router_probability=probability, token_equivalent_cost=cost,
        coverage_signature=signature, source_doc_id="doc",
    )


def _cfg(relative_floor):
    return SimpleNamespace(
        submod_alpha=1.0, submod_gamma=0.0, submod_cost_beta=0.35,
        max_total_cost=6000.0, max_groups=5, family_mutex=True,
        min_relative_probability=relative_floor,
    )


def test_submod_selector_prunes_low_confidence_tail_after_first_choice():
    groups = [
        _group("strong", 0.50, 100.0, [1.0, 0.0]),
        _group("useful", 0.10, 100.0, [0.0, 1.0]),
        _group("tail", 0.02, 10.0, [1.0, -1.0]),
    ]
    selected = SubmodKnapsackSelector(_cfg(0.05)).select(groups)
    assert [group.group_id for group in selected] == ["strong", "useful"]


def test_relative_probability_floor_can_be_disabled():
    groups = [
        _group("strong", 0.50, 100.0, [1.0, 0.0]),
        _group("tail", 0.02, 10.0, [0.0, 1.0]),
    ]
    selected = SubmodKnapsackSelector(_cfg(0.0)).select(groups)
    assert [group.group_id for group in selected] == ["strong", "tail"]


def _flat_cfg(**over):
    """The regime the 20260823 validation run actually hit.

    ~80 candidates share a nearly flat learned softmax (top probability ~0.12)
    and pages inside one document have similar coverage signatures.  With the
    old settings that combination collapsed the selection to a single group and
    left >80% of the cost budget unused.
    """
    base = dict(
        submod_alpha=1.0, submod_gamma=0.35, submod_cost_beta=0.20,
        max_total_cost=6000.0, max_groups=5, family_mutex=True,
        min_relative_probability=0.01, min_distinct_pages=4,
        confidence_mass_threshold=0.95, beta=1.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _flat_pool(pages=20):
    groups = []
    for page in range(pages):
        signature = [1.0 + 0.05 * page, 1.0 - 0.05 * page]
        for tier, node_class, cost in (("cap", "doc_summary", 121.0),
                                       ("full", "text_span", 135.0),
                                       ("shot", "pdf_page", 1560.0)):
            g = EvidenceGroup(
                group_id=f"p{page}_{tier}", root_node_id=f"p{page}_{tier}",
                node_ids=[f"p{page}_{tier}"], node_class=node_class, level=0,
                redundancy_cluster_id=f"doc:p{page}",
                router_probability=0.12 / (1.0 + 0.35 * page),
                token_equivalent_cost=cost,
                coverage_signature=signature, source_doc_id="doc")
            groups.append(g)
    return groups


def test_page_quota_keeps_selection_from_collapsing_on_a_flat_softmax():
    selected = SubmodKnapsackSelector(_flat_cfg()).select(_flat_pool())
    distinct_pages = {g.redundancy_cluster_id for g in selected}
    assert len(distinct_pages) >= 4, (
        f"expected >=4 distinct pages, got {len(distinct_pages)}: "
        f"{[g.group_id for g in selected]}")


def test_old_settings_reproduce_the_single_group_collapse():
    """Guards the diagnosis itself, so the fix cannot be silently reverted."""
    legacy = _flat_cfg(submod_gamma=0.8, submod_cost_beta=0.35,
                       min_relative_probability=0.05, min_distinct_pages=0,
                       confidence_mass_threshold=0.85)
    selected = SubmodKnapsackSelector(legacy).select(_flat_pool())
    assert len(selected) == 1


def test_page_quota_never_exceeds_the_cost_budget():
    tight = _flat_cfg(max_total_cost=400.0)
    selected = SubmodKnapsackSelector(tight).select(_flat_pool())
    assert sum(g.token_equivalent_cost for g in selected) <= 400.0


def test_family_mutex_still_holds_under_the_page_quota():
    selected = SubmodKnapsackSelector(_flat_cfg()).select(_flat_pool())
    clusters = [g.redundancy_cluster_id for g in selected]
    assert len(clusters) == len(set(clusters))


def test_greedy_selector_honours_the_same_page_quota():
    from file_router.router.selector import GreedySelector
    selected = GreedySelector(_flat_cfg()).select(_flat_pool())
    distinct_pages = {g.redundancy_cluster_id for g in selected}
    assert len(distinct_pages) >= 4


def test_relative_floor_still_prunes_once_the_page_quota_is_met():
    """The floor must keep working; the quota only defers it, never disables it."""
    groups = [
        _group("strong", 0.50, 100.0, [1.0, 0.0]),
        _group("tail", 0.001, 10.0, [0.0, 1.0]),
    ]
    cfg = _flat_cfg(min_distinct_pages=1, submod_gamma=0.0)
    selected = SubmodKnapsackSelector(cfg).select(groups)
    assert [g.group_id for g in selected] == ["strong"]


def _cost_cfg(**over):
    """Costs match campaign_20260824_035133: caption 154, fulltext 205, shot 1551."""
    base = dict(
        submod_alpha=1.0, submod_gamma=0.35, submod_cost_beta=0.20,
        max_total_cost=6000.0, max_groups=5, family_mutex=True,
        min_relative_probability=0.01, min_distinct_pages=4,
        confidence_mass_threshold=0.95, beta=1.0,
        expensive_tier_cost=800.0, expensive_tier_max_rank=1,
        expensive_tier_min_probability_share=0.3,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _tiered_pool(pages=8, shot_probability=0.001):
    groups = []
    for page in range(pages):
        for tier, node_class, cost, scale in (
                ("cap", "doc_summary", 154.0, 1.0),
                ("full", "text_span", 205.0, 0.9),
                ("shot", "pdf_page", 1551.0, shot_probability)):
            groups.append(EvidenceGroup(
                group_id=f"p{page}_{tier}", root_node_id=f"p{page}_{tier}",
                node_ids=[f"p{page}_{tier}"], node_class=node_class, level=0,
                redundancy_cluster_id=f"doc:p{page}",
                router_probability=0.2 * scale / (1.0 + 0.3 * page),
                token_equivalent_cost=cost,
                coverage_signature=[1.0 + 0.05 * page, 1.0 - 0.05 * page],
                source_doc_id="doc"))
    return groups


def test_low_confidence_expensive_tier_is_not_bought_as_quota_filler():
    """95% of screenshot spend landed on non-Gold pages in the last campaign."""
    selected = SubmodKnapsackSelector(_cost_cfg()).select(_tiered_pool())
    late_expensive = [
        g for rank, g in enumerate(selected)
        if g.token_equivalent_cost >= 800.0 and rank > 1]
    assert not late_expensive, (
        f"expensive group bought as filler: {[g.group_id for g in late_expensive]}")


def test_confident_expensive_tier_is_still_reachable():
    """A visual question must still be able to buy its page image.

    Uses pages that only offer the expensive tier, so family_mutex cannot mask
    the rule by picking a cheaper tier of the same page first.
    """
    groups = _tiered_pool(pages=4)
    for page in range(4, 7):
        groups.append(EvidenceGroup(
            group_id=f"v{page}_shot", root_node_id=f"v{page}_shot",
            node_ids=[f"v{page}_shot"], node_class="pdf_page", level=0,
            redundancy_cluster_id=f"doc:v{page}",
            router_probability=0.2,          # ties the best cheap candidate
            token_equivalent_cost=1551.0,
            coverage_signature=[0.1, 0.9], source_doc_id="doc"))
    selected = SubmodKnapsackSelector(_cost_cfg()).select(groups)
    assert any(g.token_equivalent_cost >= 800.0 for g in selected), (
        f"no image reachable: {[(g.group_id, g.token_equivalent_cost) for g in selected]}")


def test_expensive_tier_is_blocked_when_confidence_is_far_below_the_best():
    """Same shape as above but a weak image — this one must NOT be bought."""
    groups = _tiered_pool(pages=4)
    for page in range(4, 7):
        groups.append(EvidenceGroup(
            group_id=f"v{page}_shot", root_node_id=f"v{page}_shot",
            node_ids=[f"v{page}_shot"], node_class="pdf_page", level=0,
            redundancy_cluster_id=f"doc:v{page}",
            router_probability=0.001,        # far under the 0.3 share
            token_equivalent_cost=1551.0,
            coverage_signature=[0.1, 0.9], source_doc_id="doc"))
    selected = SubmodKnapsackSelector(_cost_cfg()).select(groups)
    late = [g for rank, g in enumerate(selected)
            if g.token_equivalent_cost >= 800.0 and rank > 1]
    assert not late


def test_expensive_rank_rule_can_be_disabled():
    cfg = _cost_cfg(expensive_tier_max_rank=-1,
                    expensive_tier_min_probability_share=0.0)
    groups = _tiered_pool(shot_probability=1.0)
    assert SubmodKnapsackSelector(cfg).select(groups)


def test_expensive_rule_still_respects_the_page_quota():
    selected = SubmodKnapsackSelector(_cost_cfg()).select(_tiered_pool())
    assert len({g.redundancy_cluster_id for g in selected}) >= 4


def test_expensive_rule_is_scale_free_across_routers():
    """The regression behind campaign_20260830_123023.

    Every router shares this selector, but their softmax scales differ by an
    order of magnitude (learned median 0.045 vs zeroshot max 0.019).  An
    absolute probability threshold therefore hits the cheap baseline harder than
    the learned router, making the baseline cheaper and inverting
    cost_saving_vs_zeroshot.  The rule must stay purely relative, so two pools
    that differ only by a constant scale factor must select the same tiers.
    """
    cfg = _cost_cfg(expensive_tier_min_probability_share=0.45)

    def pool(scale):
        groups = []
        for page in range(4):
            for tier, node_class, cost, weight in (
                    ("cap", "doc_summary", 154.0, 1.0),
                    ("shot", "pdf_page", 1551.0, 0.5)):
                groups.append(EvidenceGroup(
                    group_id=f"p{page}_{tier}", root_node_id=f"p{page}_{tier}",
                    node_ids=[f"p{page}_{tier}"], node_class=node_class,
                    level=0, redundancy_cluster_id=f"doc:p{page}",
                    router_probability=scale * weight / (1.0 + 0.3 * page),
                    token_equivalent_cost=cost,
                    coverage_signature=[1.0 + 0.05 * page, 1.0 - 0.05 * page],
                    source_doc_id="doc"))
        return groups

    peaked = [g.group_id for g in SubmodKnapsackSelector(cfg).select(pool(0.20))]
    flat = [g.group_id for g in SubmodKnapsackSelector(cfg).select(pool(0.01))]
    assert peaked == flat, (
        f"selection changed with probability scale alone: {peaked} vs {flat}")


def test_expensive_rule_has_no_absolute_probability_threshold():
    """Guards the invariant directly, not just its observable effect."""
    from file_router.router import selector as selector_module
    source = Path(selector_module.__file__).read_text(encoding="utf-8")
    assert "expensive_tier_min_probability\"" not in source, (
        "an absolute probability threshold was reintroduced; it breaks the "
        "zeroshot baseline (max probability ~0.019) far harder than the "
        "learned router and inverts cost_saving_vs_zeroshot")


def _ablation_pool(pages=6):
    groups = []
    for page in range(pages):
        for tier, node_class, cost in (("cap", "doc_summary", 140.0),
                                       ("full", "text_span", 215.0),
                                       ("shot", "pdf_page", 1543.0)):
            groups.append(EvidenceGroup(
                group_id=f"p{page}_{tier}", root_node_id=f"p{page}_{tier}",
                node_ids=[f"p{page}_{tier}"], node_class=node_class, level=0,
                redundancy_cluster_id=f"doc:p{page}",
                router_probability=0.2 / (1.0 + 0.3 * page),
                token_equivalent_cost=cost,
                coverage_signature=[1.0 + 0.05 * page, 1.0 - 0.05 * page],
                source_doc_id="doc"))
    return groups


def _tiers_of(selected):
    from file_router.router.selector import _tier_name
    return {_tier_name(g) for g in selected}


def test_tier_ablation_removes_the_named_tier():
    cfg = _cost_cfg(disable_tiers=["caption"])
    assert "caption" not in _tiers_of(
        SubmodKnapsackSelector(cfg).select(_ablation_pool()))


def test_tier_ablation_accepts_a_comma_separated_string():
    """So DISABLE_TIERS=caption,fulltext can be passed straight through."""
    cfg = _cost_cfg(disable_tiers="caption,fulltext")
    tiers = _tiers_of(SubmodKnapsackSelector(cfg).select(_ablation_pool()))
    assert tiers == {"screenshot"}


def test_tier_ablation_is_inert_when_unset():
    baseline = SubmodKnapsackSelector(_cost_cfg()).select(_ablation_pool())
    explicit = SubmodKnapsackSelector(
        _cost_cfg(disable_tiers=[])).select(_ablation_pool())
    assert [g.group_id for g in baseline] == [g.group_id for g in explicit]


def test_tier_ablation_never_empties_the_candidate_pool():
    """A query whose every candidate is disabled must not become no-evidence.

    Returning nothing would quietly convert that question into a different
    experiment than the one being measured.
    """
    only_captions = [g for g in _ablation_pool()
                     if g.node_class == "doc_summary"]
    cfg = _cost_cfg(disable_tiers=["caption"])
    assert SubmodKnapsackSelector(cfg).select(only_captions)


def test_tier_ablation_applies_to_the_greedy_selector_too():
    from file_router.router.selector import GreedySelector
    cfg = _cost_cfg(disable_tiers=["caption"])
    assert "caption" not in _tiers_of(
        GreedySelector(cfg).select(_ablation_pool()))
