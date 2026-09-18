"""Scorers turn EvidenceGroups into router_probability.

Two implementations:
  - RuleBasedScorer: weighted linear combination, softmax.
  - LearnedScorer: 2-layer MLP trained on past router decisions.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import List, Optional

import numpy as np

from ..schemas import EvidenceGroup, ModalityBias


# Fixed feature dimension — must match the MLP input_dim.
#
# 18 candidate/bias features + 10 inexpensive query-intent features.  The
# previous model only saw retrieval score, tier and cost, so it could mostly
# learn a global "prefer caption" policy.  Query features let it distinguish
# count/comparison/temporal/visual questions without introducing another large
# trainable encoder.
FEATURE_DIM = 28


def apply_retrieval_prior(logits, features, weight: float):
    """Add the same bounded retrieval prior in training and inference.

    RRF scores are normalized to [0, 1] by Retriever.  The bounded log curve
    preserves learned tier decisions while preventing a type prior from making
    an unrelated graph-expanded candidate outrank directly retrieved evidence.
    """
    if not weight:
        return logits
    base = features[..., 0].clamp(min=0.0, max=1.0)
    prior = base.mul(9.0).add(1.0).log().div(math.log(10.0))
    return logits + float(weight) * prior


# Keyword lists are bilingual on purpose: the corpora contain
# Chinese-language questions, so these tokens are matched against
# real input and are not translatable comments.
def build_query_features(question: str) -> List[float]:
    """Return deterministic, language-light query features.

    Keep this function shared by training and inference.  Features deliberately
    avoid dataset labels so they are also available for unseen user questions.
    """
    q = (question or "").strip().lower()
    words = re.findall(r"[\w]+", q, flags=re.UNICODE)

    def has(*terms: str) -> float:
        return float(any(term in q for term in terms))

    return [
        min(len(q) / 240.0, 1.0),
        min(len(words) / 50.0, 1.0),
        float(q.startswith(("is ", "are ", "was ", "were ", "do ", "does ",
                            "did ", "has ", "have ", "can ", "could ", "是否"))),
        has("how many", "number of", "count", "多少", "几 个", "几个"),
        has("when", "date", "year", "month", "before", "after", "何时", "时间", "哪年"),
        has("where", "which places", "list", "name all", "哪些", "哪里", "列出"),
        has("image", "photo", "figure", "chart", "diagram", "table", "icon", "logo",
            "截图", "图片", "图表", "表格", "图标"),
        has("compare", "difference", "consistent", "before and after", "versus", " vs ",
            "比较", "区别", "一致"),
        has("why", "how did", "explain", "reason", "为什么", "如何", "原因"),
        has(" and ", "both", "across", "respectively", "分别", "以及", "两者"),
    ]


def build_feature_vector(g: EvidenceGroup, bias: ModalityBias,
                         max_cost: float, question: str = "") -> List[float]:
    """Candidate + query feature vector shared by training and inference."""
    active = bias.active_modality if bias.apply_bias else "none"
    type_flags = [
        float(g.node_class == "text_span"),
        float(g.node_class == "text_cluster"),
        float(g.node_class == "doc_summary"),
        float(g.node_class == "image"),
        float(g.node_class == "pdf_page"),
        float(g.node_class == "pdf_region"),
    ]
    level_flags = [float(g.level == L) for L in (0, 1, 2, 3)]
    candidate_features = [
        float(g.base_score),
        float(g.modality_bias_score),
        float(g.token_equivalent_cost) / max(max_cost, 1.0),
        *type_flags,                # 6
        *level_flags,               # 4
        float(active == "image"),
        float(active == "pdf"),
        float(active == "text"),
        float(bias.confidence),
        float(len(g.node_ids)) / 5.0,
    ]
    return candidate_features + build_query_features(question)


class RuleBasedScorer:
    """Score = w_base·base + w_bias·bias + w_type·type_prior − w_cost·norm_cost.
    Then softmax over eligible groups with temperature scaling.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def score(self, groups: List[EvidenceGroup], max_cost: float,
              bias: Optional[ModalityBias] = None, question: str = ""):
        if not groups:
            return
        type_priors = {
            "text_span": 0.5, "text_cluster": 0.6, "doc_summary": 0.6,
            "image": 0.5, "pdf_page": 0.7, "pdf_region": 0.6,
        }
        raw = []
        for g in groups:
            norm_cost = g.token_equivalent_cost / max(max_cost, 1.0)
            g.router_score = (
                self.cfg.w_base * g.base_score
                + self.cfg.w_bias * g.modality_bias_score
                + self.cfg.w_type * type_priors.get(g.node_class, 0.4)
                - self.cfg.w_cost * norm_cost
            )
            raw.append(g.router_score)
        # softmax
        raw = np.array(raw, dtype=np.float64) / max(self.cfg.temperature, 1e-3)
        raw = raw - raw.max()
        probs = np.exp(raw)
        probs = probs / (probs.sum() + 1e-9)
        for g, p in zip(groups, probs):
            g.router_probability = float(p)


class LearnedScorer:
    """2-layer MLP; falls back to RuleBasedScorer if weights are missing."""

    def __init__(self, cfg_router, trainer_cfg, model_path: str):
        self.cfg = cfg_router
        self.model_path = model_path
        self._torch = None
        self._model = None
        self._fallback = RuleBasedScorer(cfg_router)
        self._load(trainer_cfg)

    def _load(self, trainer_cfg):
        weights = os.path.join(self.model_path, "router_weights.pt")
        meta = os.path.join(self.model_path, "router_meta.json")
        if not (os.path.exists(weights) and os.path.exists(meta)):
            print(f"[LearnedScorer] weights not found at {self.model_path}; using rule-based scorer.")
            return
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            with open(meta, "r") as f:
                m = json.load(f)
            dim = m["input_dim"]
            hidden = m["hidden_dim"]

            # Reconstruct the exact shape training used.  Checkpoints written
            # before the architecture field existed are all the shipped mlp2.
            from file_router.router.architectures import build

            arch = m.get("architecture", "mlp2")
            self._model = build(dim, hidden, arch)
            self._model.load_state_dict(torch.load(weights, map_location="cpu"))
            self._model.eval()
            print(f"[LearnedScorer] loaded {self.model_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[LearnedScorer] failed to load: {e}. Falling back.")
            self._model = None

    def score(self, groups: List[EvidenceGroup], max_cost: float,
              bias: Optional[ModalityBias] = None, question: str = ""):
        if self._model is None or self._torch is None:
            self._fallback.score(groups, max_cost, bias=bias, question=question)
            return
        if not groups:
            return
        effective_bias = bias or ModalityBias()
        feats = np.array([
            build_feature_vector(g, effective_bias, max_cost, question=question)
            for g in groups
        ], dtype=np.float32)
        with self._torch.no_grad():
            feature_tensor = self._torch.tensor(feats)
            logits = self._model(feature_tensor).squeeze(-1)
            logits = apply_retrieval_prior(
                logits, feature_tensor,
                float(getattr(self.cfg, "learned_retrieval_prior", 0.0)),
            )
            probs = self._torch.softmax(logits, dim=0).cpu().numpy()
        for index, (g, p) in enumerate(zip(groups, probs)):
            g.router_probability = float(p)
            g.router_score = float(logits[index])
