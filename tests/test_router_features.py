from types import SimpleNamespace

from file_router.router.scorer import FEATURE_DIM, build_feature_vector
from file_router.router.selector import SubmodKnapsackSelector
from file_router.schemas import EvidenceGroup, ModalityBias


def _group():
    return EvidenceGroup(
        group_id="g", root_node_id="n", node_ids=["n"],
        node_class="pdf_page", level=0, redundancy_cluster_id="d:p1",
        base_score=0.8, modality_bias_score=0.4,
        token_equivalent_cost=1200.0,
    )


def test_training_and_inference_feature_builder_preserves_real_bias():
    bias = ModalityBias("image", confidence=0.9, apply_bias=True)
    train_features = build_feature_vector(
        _group(), bias, 6000.0, question="How many icons are in Figure 1?")
    inference_features = build_feature_vector(
        _group(), bias, 6000.0, question="How many icons are in Figure 1?")
    assert train_features == inference_features
    assert len(train_features) == FEATURE_DIM == 28
    assert train_features[13] == 1.0  # active image modality
    assert train_features[16] == 0.9  # confidence
    assert train_features[21] == 1.0  # count question
    assert train_features[24] == 1.0  # visual question


def test_empty_bias_is_observably_different():
    real = build_feature_vector(_group(), ModalityBias("image", 0.9, True), 6000.0)
    empty = build_feature_vector(_group(), ModalityBias(), 6000.0)
    assert real != empty


def test_submod_selector_uses_cost_to_choose_tier_within_page():
    cheap = _group()
    cheap.group_id = "caption"
    cheap.node_class = "doc_summary"
    cheap.token_equivalent_cost = 60.0
    cheap.router_probability = 0.48
    expensive = _group()
    expensive.group_id = "screenshot"
    expensive.token_equivalent_cost = 1440.0
    expensive.router_probability = 0.50
    cfg = SimpleNamespace(
        submod_alpha=1.0, submod_gamma=0.0, submod_cost_beta=0.35,
        max_total_cost=6000.0, max_groups=5, family_mutex=True,
    )
    selected = SubmodKnapsackSelector(cfg).select([expensive, cheap])
    assert [group.group_id for group in selected] == ["caption"]
