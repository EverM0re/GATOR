"""Load router_training.jsonl into per-query tensors.

Extends cnem_v's RouterDataset with the extra targets needed by the enhanced
loss (DESIGN.md §3.3):
    labels         multi-hot over gold groups          (L_route)
    costs          normalized token-equivalent cost     (L_cost)
    minimal_labels multi-hot minimal-sufficient targets (L_level)
    cluster_ids    redundancy cluster per group         (L_margin pairing)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..router.scorer import build_feature_vector, FEATURE_DIM
from ..schemas import EvidenceGroup, ModalityBias


class RouterDataset:
    def __init__(self, path: str, use_minimal: bool, max_cost: float):
        self.path = path
        self.use_minimal = use_minimal
        self.max_cost = max_cost

    def load(self) -> List[Dict[str, Any]]:
        out = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gold_ids = set(rec.get("gold_supporting_groups") or [])
                if not gold_ids:
                    continue
                raw_minimal = rec.get("minimal_sufficient_groups")
                # None is an old/unfilled record; [] explicitly means that no
                # tier could be proven sufficient and L_level must be skipped.
                minimal_ids = set(raw_minimal or [])
                out.append({
                    "query_id": rec["query_id"],
                    "query": rec.get("query", ""),
                    "groups": rec["candidate_groups"],
                    "gold_ids": gold_ids,
                    "minimal_ids": minimal_ids,
                    "bias": rec.get("modality_bias", {}),
                })
        return out

    def to_tensors(self, records):
        import torch

        batches = []
        for rec in records:
            groups = [_dict_to_group(g) for g in rec["groups"]]
            if not groups:
                continue
            gold_ids = rec["gold_ids"]
            minimal_ids = rec["minimal_ids"]
            bias_d = rec["bias"]
            bias = ModalityBias(active_modality=bias_d.get("active_modality", "none"),
                                confidence=float(bias_d.get("confidence", 0.0)),
                                apply_bias=bool(bias_d.get("apply_bias", False)))

            features = torch.tensor(
                [build_feature_vector(g, bias, self.max_cost,
                                      question=rec.get("query", ""))
                 for g in groups],
                dtype=torch.float32)
            assert features.shape[1] == FEATURE_DIM, \
                f"feature dim {features.shape[1]} != {FEATURE_DIM}"
            labels = torch.tensor(
                [1.0 if g.group_id in gold_ids else 0.0 for g in groups],
                dtype=torch.float32)
            costs = torch.tensor(
                [g.token_equivalent_cost / max(self.max_cost, 1.0) for g in groups],
                dtype=torch.float32)
            minimal_labels = torch.tensor(
                [1.0 if g.group_id in minimal_ids else 0.0 for g in groups],
                dtype=torch.float32)
            # integer cluster ids for L_margin pairing
            uniq = {}
            cluster_ids = []
            for g in groups:
                cid = uniq.setdefault(g.redundancy_cluster_id, len(uniq))
                cluster_ids.append(cid)

            batches.append({
                "features": features,
                "labels": labels,
                "costs": costs,
                "minimal_labels": minimal_labels,
                "cluster_ids": torch.tensor(cluster_ids, dtype=torch.long),
                "raw_costs": torch.tensor([g.token_equivalent_cost for g in groups],
                                          dtype=torch.float32),
                "query_id": rec["query_id"],
            })
        return batches


def _dict_to_group(d: dict) -> EvidenceGroup:
    return EvidenceGroup(
        group_id=d["group_id"], root_node_id=d["root_node_id"],
        node_ids=d["node_ids"], node_class=d["node_class"], level=d["level"],
        redundancy_cluster_id=d["redundancy_cluster_id"],
        base_score=d.get("base_score", 0.0),
        modality_bias_score=d.get("modality_bias_score", 0.0),
        router_score=d.get("router_score", 0.0),
        router_probability=d.get("router_probability", 0.0),
        token_equivalent_cost=d.get("token_equivalent_cost", 0.0),
        coverage_signature=d.get("coverage_signature"),
        source_doc_id=d.get("source_doc_id", ""),
    )
