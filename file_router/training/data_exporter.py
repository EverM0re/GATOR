"""Writes one JSONL record per inference to the training data path.

Each record is a full snapshot of the router decision, with `gold_*` labels
initially null. Use `fill_gold_labels(query_id, ...)` later to supervise.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import List, Optional

from ..schemas import EvidenceGroup, ModalityBias


class DataExporter:
    def __init__(self, path: str, compact: bool = False):
        self.path = path
        self.compact = compact
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # -------------------------------------------------------------- export
    def export(self, question: str, answer: str,
               candidate_groups: List[EvidenceGroup],
               selected_groups: List[EvidenceGroup],
               bias: ModalityBias,
               query_id: Optional[str] = None,
               gold_groups: Optional[List[str]] = None,
               minimal_groups: Optional[List[str]] = None,
               alternative_level_solutions: Optional[dict] = None,
               correct: Optional[bool] = None) -> str:
        query_id = query_id or f"q_{uuid.uuid4().hex[:10]}"
        record = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": question,
            "answer": answer,
            "modality_bias": {
                "active_modality": bias.active_modality,
                "confidence": bias.confidence,
                "apply_bias": bias.apply_bias,
            },
            "candidate_groups": [self._group_to_dict(g) for g in candidate_groups],
            "selected_group_ids": [g.group_id for g in selected_groups],
            "gold_supporting_groups": gold_groups,
            "minimal_sufficient_groups": minimal_groups,
            "alternative_level_solutions": alternative_level_solutions,
            "final_answer_correct": correct,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return query_id

    # -------------------------------------------------------------- labels
    def fill_gold_labels(self, query_id: str,
                         gold_groups: List[str],
                         minimal_groups: Optional[List[str]] = None,
                         alternative_level_solutions: Optional[dict] = None,
                         correct: Optional[bool] = None) -> bool:
        """Reload the JSONL, patch the matching record, rewrite.
        Returns True if the record was found."""
        if not os.path.exists(self.path):
            return False
        out = []
        patched = False
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("query_id") == query_id:
                    rec["gold_supporting_groups"] = gold_groups
                    if minimal_groups is not None:
                        rec["minimal_sufficient_groups"] = minimal_groups
                    if alternative_level_solutions is not None:
                        rec["alternative_level_solutions"] = alternative_level_solutions
                    if correct is not None:
                        rec["final_answer_correct"] = correct
                    patched = True
                out.append(json.dumps(rec, ensure_ascii=False))
        if patched:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(out) + "\n")
        return patched

    # ------------------------------------------------------------ helpers
    def _group_to_dict(self, g: EvidenceGroup) -> dict:
        row = {
            "group_id": g.group_id,
            "root_node_id": g.root_node_id,
            "node_ids": g.node_ids,
            "node_class": g.node_class,
            "level": g.level,
            "redundancy_cluster_id": g.redundancy_cluster_id,
            "base_score": g.base_score,
            "modality_bias_score": g.modality_bias_score,
            "router_score": g.router_score,
            "router_probability": g.router_probability,
            "token_equivalent_cost": g.token_equivalent_cost,
            "source_doc_id": g.source_doc_id,
        }
        # Coverage signatures dominate the JSON size (64 floats per candidate)
        # but are never used by RouterDataset.  Keep full inference snapshots
        # compatible while making collected training files roughly 10x smaller.
        if not self.compact:
            row["coverage_signature"] = g.coverage_signature
        return row
