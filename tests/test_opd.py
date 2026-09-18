from types import SimpleNamespace

import torch

from file_router.training.opd import (
    cost_aware_teacher_logits,
    sequential_opd_loss,
)
from scripts import train


def _opd_cfg(**overrides):
    values = {
        "enabled": True,
        "run_ab": True,
        "divergence": "reverse_kl",
        "rollout_steps": 1,
        "student_temperature": 1.0,
        "teacher_temperature": 1.0,
        "teacher_gold_bonus": 2.0,
        "teacher_minimal_bonus": 1.5,
        "teacher_retrieval_weight": 0.5,
        "teacher_cost_weight": 0.75,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _inputs():
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    minimal = torch.tensor([[1.0, 0.0, 0.0]])
    costs = torch.tensor([[0.1, 0.2, 0.3]])
    raw_costs = torch.tensor([[10.0, 20.0, 30.0]])
    clusters = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[True, True, True]])
    retrieval = torch.tensor([[0.9, 0.8, 0.7]])
    return labels, minimal, costs, raw_costs, clusters, mask, retrieval


def test_opd_loss_is_small_when_student_matches_teacher():
    values = _inputs()
    cfg = _opd_cfg()
    teacher = cost_aware_teacher_logits(
        values[0], values[1], values[2], values[6], cfg)
    aligned, _ = sequential_opd_loss(
        teacher.clone().requires_grad_(True), *values[:6], values[6], cfg,
        max_total_cost=100.0, family_mutex=True, sample_actions=False)
    reversed_loss, _ = sequential_opd_loss(
        (-teacher).requires_grad_(True), *values[:6], values[6], cfg,
        max_total_cost=100.0, family_mutex=True, sample_actions=False)
    assert aligned.item() < 1e-6
    assert reversed_loss.item() > aligned.item() + 0.1


def test_opd_rollout_visits_student_partial_selection_states():
    labels, minimal, costs, raw_costs, _clusters, mask, retrieval = _inputs()
    clusters = torch.tensor([[0, 0, 1]])
    logits = torch.tensor([[5.0, 4.0, 3.0]], requires_grad=True)
    loss, stats = sequential_opd_loss(
        logits, labels, minimal, costs, raw_costs, clusters, mask, retrieval,
        _opd_cfg(rollout_steps=5), max_total_cost=100.0,
        family_mutex=True, sample_actions=False)
    loss.backward()
    assert stats["states"] == 2
    assert torch.isfinite(logits.grad).all()


def test_train_stage_builds_same_seed_baseline_and_opd_models(monkeypatch):
    calls = []

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self, data_path, model_dir):
            calls.append((self.cfg.trainer.opd.enabled, data_path, model_dir))

    monkeypatch.setattr(train, "RouterTrainer", FakeTrainer)
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(opd=_opd_cfg()),
        paths=SimpleNamespace(
            training_data_path="train.jsonl", router_model_dir="model"),
    )
    assert train.fit_router_models(cfg) == ["model_baseline", "model"]
    assert calls == [
        (False, "train.jsonl", "model_baseline"),
        (True, "train.jsonl", "model"),
    ]
