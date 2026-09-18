"""Train the 2-layer MLP router scorer with the cost-ladder loss (DESIGN.md §3.3).

    L = λ_route·L_route + λ_cost·L_cost + λ_level·L_level
        + λ_margin·L_margin + λ_opd·L_opd

    L_route  = -log Σ_{g∈gold} softmax(logits)_g            # pick answer-supporting evidence
    L_cost   = Σ_i softmax(logits)_i · cost_i                # expected cost → prefer cheap
    L_level  = cross-entropy against all minimal-sufficient targets
    L_margin = Σ over same-page (minimal, more-expensive gold) pairs:
                  max(0, m − (logit_a − logit_b))            # cheaper-gold must outrank pricier
    L_opd    = policy divergence to a cost-aware oracle teacher on
               sequential partial selections sampled from the student

The MLP architecture matches LearnedScorer.MLP so weights load directly.
"""

from __future__ import annotations

import json
import copy
import hashlib
import os
import random
import time

import numpy as np

from .dataset import RouterDataset
from .opd import sequential_opd_loss
from ..router.scorer import apply_retrieval_prior
from ..utils import dbg, info


class RouterTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self, data_path: str, save_dir: str):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        tc = self.cfg.trainer
        random.seed(tc.seed); np.random.seed(tc.seed); torch.manual_seed(tc.seed)

        ds = RouterDataset(data_path, use_minimal=tc.use_minimal_sufficient,
                           max_cost=self.cfg.cost.max_cost_normalizer)
        records = ds.load()
        if not records:
            raise RuntimeError(
                f"No labeled records in {data_path}. Run train.py collect stage first.")
        batches = ds.to_tensors(records)
        info(f"[trainer] {len(batches)} labeled queries from {data_path}")

        # Built from the shared definition so inference reconstructs exactly
        # this shape; the architecture name is persisted in router_meta.json.
        from file_router.router.architectures import build, parameter_count

        arch = getattr(tc, "architecture", "mlp2")
        device = tc.device
        model = build(tc.input_dim, tc.hidden_dim, arch).to(device)
        info(f"[trainer] architecture={arch} "
             f"params={parameter_count(tc.input_dim, tc.hidden_dim, arch)}")
        opt = optim.Adam(model.parameters(), lr=tc.learning_rate)
        margin = getattr(tc, "margin", 0.5)
        lam_margin = getattr(tc, "lambda_margin", 0.0)
        lam_opd = float(getattr(tc, "lambda_opd", 0.0))
        opd_cfg = getattr(tc, "opd", None)
        opd_enabled = bool(getattr(opd_cfg, "enabled", False)) and lam_opd > 0
        val_ratio = float(getattr(tc, "validation_ratio", 0.0))
        patience = int(getattr(tc, "early_stopping_patience", 0))
        min_delta = float(getattr(tc, "early_stopping_min_delta", 1e-4))
        mini_batch_size = max(1, int(getattr(tc, "batch_size", 32)))
        retrieval_prior = float(getattr(
            self.cfg.router, "learned_retrieval_prior", 0.0))

        val_batches = []
        train_batches = list(batches)
        if val_ratio > 0 and len(batches) >= 10:
            ranked = sorted(
                batches,
                key=lambda batch: hashlib.sha256(
                    f"{tc.seed}:{batch['query_id']}".encode("utf-8")).hexdigest(),
            )
            val_count = max(1, min(len(ranked) - 1,
                                   round(len(ranked) * val_ratio)))
            val_batches = ranked[:val_count]
            train_batches = ranked[val_count:]
        info(f"[trainer] train={len(train_batches)} validation={len(val_batches)}")

        def _collate(query_batches):
            count = len(query_batches)
            max_candidates = max(batch["features"].shape[0]
                                 for batch in query_batches)
            dim = query_batches[0]["features"].shape[1]
            features = torch.zeros(count, max_candidates, dim)
            labels = torch.zeros(count, max_candidates)
            costs = torch.zeros(count, max_candidates)
            minimal_labels = torch.zeros(count, max_candidates)
            cluster_ids = torch.full((count, max_candidates), -1, dtype=torch.long)
            raw_costs = torch.zeros(count, max_candidates)
            mask = torch.zeros(count, max_candidates, dtype=torch.bool)
            for row, batch in enumerate(query_batches):
                size = batch["features"].shape[0]
                features[row, :size] = batch["features"]
                labels[row, :size] = batch["labels"]
                costs[row, :size] = batch["costs"]
                minimal_labels[row, :size] = batch["minimal_labels"]
                cluster_ids[row, :size] = batch["cluster_ids"]
                raw_costs[row, :size] = batch["raw_costs"]
                mask[row, :size] = True
            return {
                "features": features.to(device), "labels": labels.to(device),
                "costs": costs.to(device),
                "minimal_labels": minimal_labels.to(device),
                "cluster_ids": cluster_ids.to(device),
                "raw_costs": raw_costs.to(device), "mask": mask.to(device),
            }

        def batch_loss(query_batches, sample_opd: bool = False):
            batch = _collate(query_batches)
            feats, mask = batch["features"], batch["mask"]
            labels, costs = batch["labels"], batch["costs"]
            minimal_labels = batch["minimal_labels"]
            clusters, raw_costs = batch["cluster_ids"], batch["raw_costs"]

            logits = model(feats).squeeze(-1)
            logits = apply_retrieval_prior(
                logits, feats, retrieval_prior).masked_fill(~mask, -1e9)
            probs = torch.softmax(logits, dim=1)
            pos_mass = (probs * labels).sum(dim=1).clamp(min=1e-9)
            l_route = -torch.log(pos_mass).mean()
            l_cost = (probs * costs).sum(dim=1).mean()

            minimal_count = minimal_labels.sum(dim=1)
            has_minimal = minimal_count > 0
            minimal_target = minimal_labels / minimal_count.clamp(min=1.0)[:, None]
            level_each = -(minimal_target * torch.log_softmax(logits, dim=1)).sum(dim=1)
            l_level = (level_each[has_minimal].mean() if has_minimal.any()
                       else torch.tensor(0.0, device=device))

            l_margin = torch.tensor(0.0, device=device)
            if lam_margin > 0:
                # [B, N, N]: i is a minimal tier, j is a more expensive gold
                # tier in the same page cluster.  This replaces the old nested
                # Python loop and runs efficiently as one GPU operation.
                pair_mask = (
                    (minimal_labels > 0.5)[:, :, None]
                    & (labels > 0.5)[:, None, :]
                    & mask[:, :, None] & mask[:, None, :]
                    & (clusters[:, :, None] == clusters[:, None, :])
                    & (raw_costs[:, :, None] < raw_costs[:, None, :])
                )
                pair_count = pair_mask.sum(dim=(1, 2))
                has_pairs = pair_count > 0
                if has_pairs.any():
                    logit_gap = logits[:, :, None] - logits[:, None, :]
                    margin_values = torch.relu(margin - logit_gap) * pair_mask
                    margin_each = margin_values.sum(dim=(1, 2)) / pair_count.clamp(min=1)
                    l_margin = margin_each[has_pairs].mean()

            l_opd, opd_stats = sequential_opd_loss(
                logits, labels, minimal_labels, costs, raw_costs, clusters,
                mask, feats[..., 0], opd_cfg,
                max_total_cost=float(getattr(
                    self.cfg.router, "max_total_cost", 6000.0)),
                family_mutex=bool(getattr(
                    self.cfg.router, "family_mutex", True)),
                sample_actions=sample_opd,
            )

            loss = (tc.lambda_route * l_route
                    + tc.lambda_cost * l_cost
                    + tc.lambda_level * l_level
                    + lam_margin * l_margin
                    + lam_opd * l_opd)
            return (loss, l_route, l_cost, l_level, l_margin, l_opd,
                    opd_stats)

        best_state = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        best_epoch = 0
        stale_epochs = 0
        # Per-epoch history, so convergence can be shown from the saved run
        # rather than reconstructed by parsing log text.
        history = []
        for epoch in range(tc.num_epochs):
            epoch_started = time.perf_counter()
            model.train()
            random.shuffle(train_batches)
            agg = {"loss": 0.0, "route": 0.0, "cost": 0.0,
                   "level": 0.0, "margin": 0.0, "opd": 0.0,
                   "grad_norm": 0.0, "opd_states": 0, "opd_agree": 0}
            seen = 0
            for start in range(0, len(train_batches), mini_batch_size):
                query_batch = train_batches[start:start + mini_batch_size]
                (loss, l_route, l_cost, l_level, l_margin, l_opd,
                 _opd_stats) = batch_loss(
                    query_batch, sample_opd=opd_enabled)
                opt.zero_grad(); loss.backward(); opt.step()
                weight = len(query_batch)
                seen += weight
                # Teacher agreement is the diagnostic that says whether
                # distillation is still learning after the loss has flattened.
                agg["opd_states"] += _opd_stats.get("states", 0)
                agg["opd_agree"] += _opd_stats.get("top1_agreements", 0)
                # Gradient norm decaying toward zero is independent evidence of
                # convergence that does not depend on the loss scale.
                agg["grad_norm"] += float(sum(
                    float(p.grad.norm()) ** 2 for p in model.parameters()
                    if p.grad is not None) ** 0.5) * weight
                agg["loss"] += float(loss.item()) * weight
                agg["route"] += float(l_route.item()) * weight
                agg["cost"] += float(l_cost.item()) * weight
                agg["level"] += float(l_level.item()) * weight
                agg["margin"] += float(l_margin.item()) * weight
                agg["opd"] += float(l_opd.item()) * weight

            nb = max(seen, 1)
            val_loss = None
            if val_batches:
                model.eval()
                with torch.no_grad():
                    val_total = 0.0
                    for start in range(0, len(val_batches), mini_batch_size):
                        query_batch = val_batches[start:start + mini_batch_size]
                        val_total += (float(batch_loss(
                            query_batch, sample_opd=False)[0].item()) *
                                      len(query_batch))
                    val_loss = val_total / len(val_batches)
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            else:
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            elapsed = time.perf_counter() - epoch_started
            if val_loss is not None:
                info(f"[trainer] epoch {epoch+1}/{tc.num_epochs} "
                     f"loss={agg['loss']/nb:.4f} val={val_loss:.4f} "
                     f"seconds={elapsed:.1f}")
            else:
                info(f"[trainer] epoch {epoch+1}/{tc.num_epochs} "
                     f"loss={agg['loss']/nb:.4f} seconds={elapsed:.1f}")
            history.append({
                "epoch": epoch + 1,
                "train_loss": agg["loss"] / nb,
                "val_loss": val_loss,
                # Per-term losses, so a flat total can be checked for a
                # trade-off between terms rather than assumed to be a plateau.
                "route_loss": agg["route"] / nb,
                "cost_loss": agg["cost"] / nb,
                "level_loss": agg["level"] / nb,
                "margin_loss": agg["margin"] / nb,
                "opd_loss": agg["opd"] / nb,
                "grad_norm": agg["grad_norm"] / nb,
                "opd_top1_agreement": (agg["opd_agree"] / agg["opd_states"]
                                       if agg["opd_states"] else None),
                "learning_rate": opt.param_groups[0]["lr"],
            })
            dbg(f"[trainer] epoch {epoch+1} route={agg['route']/nb:.4f} "
                f"cost={agg['cost']/nb:.4f} level={agg['level']/nb:.4f} "
                f"margin={agg['margin']/nb:.4f} opd={agg['opd']/nb:.4f}")
            if val_batches and patience > 0 and stale_epochs >= patience:
                info(f"[trainer] early stop at epoch {epoch+1}; "
                     f"best_epoch={best_epoch} best_val={best_val:.4f}")
                break

        model.load_state_dict(best_state)

        heldout_metrics = {}
        metric_batches = val_batches or train_batches
        if metric_batches:
            model.eval()
            gold_hits = 0
            page_hits = 0
            minimal_hits = 0
            minimal_queries = 0
            top_costs = []
            opd_kl_weighted = 0.0
            opd_state_count = 0
            opd_agreements = 0.0
            with torch.no_grad():
                for start in range(0, len(metric_batches), mini_batch_size):
                    query_batch = metric_batches[start:start + mini_batch_size]
                    packed = _collate(query_batch)
                    logits = model(packed["features"]).squeeze(-1)
                    logits = apply_retrieval_prior(
                        logits, packed["features"], retrieval_prior,
                    ).masked_fill(~packed["mask"], -1e9)
                    l_opd, opd_stats = sequential_opd_loss(
                        logits, packed["labels"], packed["minimal_labels"],
                        packed["costs"], packed["raw_costs"],
                        packed["cluster_ids"], packed["mask"],
                        packed["features"][..., 0], opd_cfg,
                        max_total_cost=float(getattr(
                            self.cfg.router, "max_total_cost", 6000.0)),
                        family_mutex=bool(getattr(
                            self.cfg.router, "family_mutex", True)),
                        sample_actions=False,
                    )
                    opd_kl_weighted += float(l_opd.item()) * opd_stats["states"]
                    opd_state_count += opd_stats["states"]
                    opd_agreements += opd_stats["top1_agreements"]
                    top = logits.argmax(dim=1)
                    row_index = torch.arange(len(query_batch), device=device)
                    gold_hits += int(
                        packed["labels"][row_index, top].sum().item())
                    # Does the top-1 land on a gold PAGE (any tier)?  This is
                    # the training-time proxy for the end-to-end
                    # gold_page_any_selection_rate the quality gate checks.
                    top_cluster = packed["cluster_ids"][row_index, top]
                    same_page = (packed["cluster_ids"] ==
                                 top_cluster[:, None]) & packed["mask"]
                    page_hits += int(
                        ((packed["labels"] * same_page).sum(dim=1) > 0)
                        .sum().item())
                    has_minimal = packed["minimal_labels"].sum(dim=1) > 0
                    minimal_queries += int(has_minimal.sum().item())
                    if has_minimal.any():
                        minimal_hits += int(
                            packed["minimal_labels"][row_index, top][has_minimal]
                            .sum().item())
                    top_costs.extend(
                        packed["raw_costs"][row_index, top].cpu().tolist())
            heldout_metrics = {
                "source": "validation" if val_batches else "training",
                "queries": len(metric_batches),
                "top1_gold_accuracy": gold_hits / len(metric_batches),
                "top1_gold_page_accuracy": page_hits / len(metric_batches),
                "top1_minimal_accuracy": (
                    minimal_hits / minimal_queries if minimal_queries else None),
                "avg_top1_cost": float(np.mean(top_costs)) if top_costs else None,
                "opd_teacher_kl": (
                    opd_kl_weighted / opd_state_count if opd_state_count else None),
                "opd_teacher_top1_agreement": (
                    opd_agreements / opd_state_count if opd_state_count else None),
                "opd_visited_states": opd_state_count,
            }

        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "router_weights.pt"))
        with open(os.path.join(save_dir, "router_meta.json"), "w") as f:
            json.dump({"input_dim": tc.input_dim, "hidden_dim": tc.hidden_dim,
                       "architecture": arch,
                       "lambda_route": tc.lambda_route, "lambda_cost": tc.lambda_cost,
                       "lambda_level": tc.lambda_level,
                       "lambda_margin": lam_margin,
                       "lambda_opd": lam_opd,
                       "opd": {
                           "enabled": opd_enabled,
                           "run_ab": bool(getattr(opd_cfg, "run_ab", False)),
                           "divergence": getattr(opd_cfg, "divergence", None),
                           "rollout_steps": getattr(opd_cfg, "rollout_steps", None),
                           "student_temperature": getattr(
                               opd_cfg, "student_temperature", None),
                           "teacher_temperature": getattr(
                               opd_cfg, "teacher_temperature", None),
                           "teacher_gold_bonus": getattr(
                               opd_cfg, "teacher_gold_bonus", None),
                           "teacher_minimal_bonus": getattr(
                               opd_cfg, "teacher_minimal_bonus", None),
                           "teacher_retrieval_weight": getattr(
                               opd_cfg, "teacher_retrieval_weight", None),
                           "teacher_cost_weight": getattr(
                               opd_cfg, "teacher_cost_weight", None),
                       },
                       "batch_size": mini_batch_size,
                       "learned_retrieval_prior": retrieval_prior,
                       "selector_min_relative_probability": float(getattr(
                           self.cfg.router, "min_relative_probability", 0.05)),
                       "validation_ratio": val_ratio,
                       "best_epoch": best_epoch,
                       "best_validation_loss": (best_val if val_batches else None),
                       "epochs_run": len(history),
                       # True only if patience terminated the run; False means
                       # the epoch budget did, which is not evidence of
                       # convergence.
                       "early_stopped": bool(
                           val_batches and patience > 0
                           and len(history) < tc.num_epochs),
                       "history": history,
                       "heldout_metrics": heldout_metrics},
                      f, indent=2)
        info(f"[trainer] saved model -> {save_dir}")
