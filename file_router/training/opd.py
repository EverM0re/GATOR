"""On-policy distillation objective for sequential evidence selection.

Classic OPD trains an autoregressive student on states visited by its own
rollouts. File_Router's trainable component is instead a small policy over
evidence groups. The analogous state is the partially selected evidence set.
At each step we sample from the current router, apply the real family/budget
constraints, and match a cost-aware oracle teacher on the student-visited
states.

This is an auxiliary router loss, not token-level LLM OPD. The answer VLM
remains an inference-only OpenAI-compatible endpoint.
"""

from __future__ import annotations


def _cfg_value(cfg, name: str, default):
    return getattr(cfg, name, default) if cfg is not None else default


def cost_aware_teacher_logits(labels, minimal_labels, costs, retrieval_scores,
                              cfg):
    """Construct a dense teacher policy over the current candidate actions."""
    gold_bonus = float(_cfg_value(cfg, "teacher_gold_bonus", 2.0))
    minimal_bonus = float(_cfg_value(cfg, "teacher_minimal_bonus", 1.5))
    retrieval_weight = float(_cfg_value(cfg, "teacher_retrieval_weight", 0.5))
    cost_weight = float(_cfg_value(cfg, "teacher_cost_weight", 0.75))
    return (
        gold_bonus * labels
        + minimal_bonus * minimal_labels
        + retrieval_weight * retrieval_scores.clamp(0.0, 1.0)
        - cost_weight * costs.clamp(min=0.0)
    )


def sequential_opd_loss(
    logits,
    labels,
    minimal_labels,
    costs,
    raw_costs,
    cluster_ids,
    mask,
    retrieval_scores,
    cfg,
    *,
    max_total_cost: float,
    family_mutex: bool,
    sample_actions: bool,
):
    """Match the teacher on partial selections reached by the student policy.

    Sampling is stop-gradient, as in standard on-policy distillation. The
    action space is small enough to compute an exact KL at each visited state;
    only the sequence of states is sampled. Validation uses greedy actions to
    make early stopping deterministic.
    """
    import torch

    zero = logits.sum() * 0.0
    if cfg is None or not bool(_cfg_value(cfg, "enabled", False)):
        return zero, {"states": 0, "top1_agreements": 0.0}

    steps = max(1, int(_cfg_value(cfg, "rollout_steps", 5)))
    student_temperature = max(
        1e-4, float(_cfg_value(cfg, "student_temperature", 1.0)))
    teacher_temperature = max(
        1e-4, float(_cfg_value(cfg, "teacher_temperature", 0.5)))
    divergence = str(_cfg_value(cfg, "divergence", "reverse_kl")).lower()
    if divergence not in {"reverse_kl", "forward_kl"}:
        raise ValueError(
            "trainer.opd.divergence must be 'reverse_kl' or 'forward_kl'")

    teacher_logits = cost_aware_teacher_logits(
        labels, minimal_labels, costs, retrieval_scores, cfg)
    feasible = mask.clone()
    budget = float(max_total_cost)
    if budget > 0:
        within_budget = raw_costs <= budget
        initial = feasible & within_budget
        no_budget_action = ~initial.any(dim=1)
        # Inference falls back to top-1 if every item exceeds the budget.
        initial[no_budget_action] = feasible[no_budget_action]
        feasible = initial

    spent = torch.zeros(logits.shape[0], device=logits.device,
                        dtype=raw_costs.dtype)
    total_kl = zero
    state_count = 0
    agreement_count = 0.0

    for _ in range(steps):
        active = feasible.any(dim=1)
        if not bool(active.any()):
            break

        safe_feasible = feasible.clone()
        # Avoid an all-masked softmax for inactive rows; those rows are omitted
        # from every loss and diagnostic below.
        safe_feasible[~active, 0] = True
        student_logp = torch.log_softmax(
            (logits / student_temperature).masked_fill(~safe_feasible, -1e9),
            dim=1,
        )
        teacher_logp = torch.log_softmax(
            (teacher_logits / teacher_temperature).masked_fill(
                ~safe_feasible, -1e9),
            dim=1,
        )
        student_prob = student_logp.exp()
        teacher_prob = teacher_logp.exp()
        if divergence == "forward_kl":
            kl_each = (teacher_prob * (teacher_logp - student_logp)).sum(dim=1)
        else:
            kl_each = (student_prob * (student_logp - teacher_logp)).sum(dim=1)
        total_kl = total_kl + kl_each[active].sum()
        state_count += int(active.sum().item())
        agreement_count += float(
            (student_logp.argmax(dim=1)[active]
             == teacher_logp.argmax(dim=1)[active]).sum().item())

        if sample_actions:
            action = torch.multinomial(student_prob.detach(), 1).squeeze(1)
        else:
            action = student_logp.detach().argmax(dim=1)

        row = torch.arange(logits.shape[0], device=logits.device)
        chosen_cost = raw_costs[row, action]
        spent = spent + torch.where(active, chosen_cost, torch.zeros_like(chosen_cost))
        chosen_cluster = cluster_ids[row, action]

        if family_mutex:
            remove = cluster_ids == chosen_cluster[:, None]
        else:
            remove = torch.zeros_like(feasible)
            remove[row, action] = True
        feasible = feasible & ~(remove & active[:, None])
        if budget > 0:
            feasible = feasible & ((spent[:, None] + raw_costs) <= budget)

    if state_count == 0:
        return zero, {"states": 0, "top1_agreements": 0.0}
    return total_kl / state_count, {
        "states": state_count,
        "top1_agreements": agreement_count,
    }
