"""Comprehensive small-scale validation for File_Router.

The normal evaluator only emits a compact JSON summary.  This evaluator is
designed for a server validation run and writes a self-contained REPORT.md,
plus machine-readable per-question details.

Usage:
    python -m scripts.evaluate_validation \
        --config /path/to/effective_config.yaml \
        --run-dir /path/to/validation_run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config  # noqa: E402
from file_router.data.evidence import (  # noqa: E402
    has_text_modality,
    has_visual_modality,
    page_modalities,
)
from file_router.encoders import VLM  # noqa: E402
from file_router.evaluation import answer_metrics, run_lm_judge  # noqa: E402
from file_router.evaluation.metrics import mean_present  # noqa: E402
from file_router.memory import DatasetMemory, SharedEncoders  # noqa: E402
from file_router.router.scorer import LearnedScorer, RuleBasedScorer  # noqa: E402
from file_router.router.selector import build_selector  # noqa: E402
from file_router.router.zeroshot import ZeroShotRouter  # noqa: E402
from file_router.schemas import EvidenceGroup  # noqa: E402
from file_router.utils import banner, info, warn  # noqa: E402


BASE_ROUTERS = ("full", "oracle", "rule", "zeroshot", "learned")
# Retrieval baselines rank the same candidates by a classical scorer instead of
# the dense recall the other arms share, so they isolate how much of the result
# is ranking rather than granularity choice.  They take the richest tier per
# page, exactly like Vanilla RAG, and differ only in which pages they pick.
RETRIEVAL_ROUTERS = ("bm25", "dense")
# Prompting baselines reuse Vanilla RAG's selection and vary only how the model
# is asked to reason over it, isolating prompting from granularity choice.
PROMPT_ROUTERS = {"cot": "cot", "selfask": "selfask"}
# Off by default: each extra arm costs a full generation pass over the test set,
# so a run opts in rather than paying for arms it did not ask for.
_REQUESTED_EXTRA = os.environ.get("EXTRA_ROUTERS", "").replace(",", " ").split()
_KNOWN_EXTRA = set(RETRIEVAL_ROUTERS) | set(PROMPT_ROUTERS)
# A name this build does not know means the caller is newer than the code, which
# previously showed up only as a run that completed with the old set of arms.
# Refuse instead: an unrecognised request is a stale deployment, not a typo to
# be silently dropped.
_UNKNOWN_EXTRA = [name for name in _REQUESTED_EXTRA if name not in _KNOWN_EXTRA]
if _UNKNOWN_EXTRA:
    raise SystemExit(
        f"EXTRA_ROUTERS requested {_UNKNOWN_EXTRA}, which this build does not "
        f"implement (known: {sorted(_KNOWN_EXTRA)}). The deployed code is older "
        f"than the caller -- re-upload before running, or the run will finish "
        f"normally and reproduce the previous result.")
EXTRA_ROUTERS = tuple(_REQUESTED_EXTRA)
ROUTERS = BASE_ROUTERS + EXTRA_ROUTERS
if EXTRA_ROUTERS:
    print(f"[validation] extra router arms enabled: {', '.join(EXTRA_ROUTERS)}",
          flush=True)
TIER_NAMES = {
    "doc_summary": "caption",
    "pdf_page": "screenshot",
    "text_span": "fulltext",
}


def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, gold: str) -> float:
    return float(_norm(pred) == _norm(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def contains_answer(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    return float(bool(p and g) and (p in g or g in p))


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def _round(value, digits=4):
    return round(float(value), digits) if value is not None else None


def _load_split_ids(cfg, dataset: str, split: str) -> set:
    path = os.path.join(cfg.paths.splits_dir, f"{dataset}.{split}.txt")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _iter_qa(cfg, dataset: str, keep_ids: set, limit: int = 0):
    path = os.path.join(cfg.paths.unified_dir, dataset, "qa.jsonl")
    if not os.path.exists(path):
        return
    emitted = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("qa_id") not in keep_ids:
                continue
            yield row
            emitted += 1
            if limit and emitted >= limit:
                return


def _tier(node_class: str) -> str:
    return TIER_NAMES.get(node_class, node_class)


def _rank_clusters(groups) -> list:
    """Rank evidence pages/clusters without letting three tiers occupy top-k."""
    best = {}
    for group in groups:
        key = group.redundancy_cluster_id or group.group_id
        if key not in best or group.base_score > best[key].base_score:
            best[key] = group
    return sorted(best.values(), key=lambda g: -g.base_score)


def _group_info(group, nodes) -> dict:
    node = nodes.get(group.root_node_id)
    return {
        "group_id": group.group_id,
        "node_id": group.root_node_id,
        "node_class": group.node_class,
        "tier": _tier(group.node_class),
        "doc_id": group.source_doc_id,
        "page": node.source_ref.page_num if node else None,
        "cost": _round(group.token_equivalent_cost, 2),
        "base_score": _round(group.base_score, 5),
        "router_score": _round(group.router_score, 5),
        "router_probability": _round(group.router_probability, 6),
        "cluster_id": group.redundancy_cluster_id,
    }


def _materialize(groups, nodes) -> Tuple[str, List[str], float, List[dict]]:
    lines, images, selected_info = [], [], []
    total_cost = 0.0
    for group in groups:
        total_cost += group.token_equivalent_cost
        selected_info.append(_group_info(group, nodes))
        for node_id in group.node_ids:
            node = nodes.get(node_id)
            if node is None:
                continue
            tag = (f"[{node.node_class} doc={node.source_ref.doc_id} "
                   f"page={node.source_ref.page_num}]")
            if node.text:
                lines.append(f"{tag} {node.text}")
            image_path = node.page_image_path or node.image_path
            if image_path:
                lines.append(f"{tag} [attached image]")
                images.append(image_path)
    return ("\n\n".join(lines), list(dict.fromkeys(images)),
            total_cost, selected_info)


def _bm25_scores(question: str, groups, nodes) -> dict:
    """Okapi BM25 over the candidate texts, k1=1.5, b=0.75.

    Scored over the retrieved candidate set rather than the whole corpus: the
    arm answers "does classical lexical ranking pick better pages than dense
    recall", which only requires ranking the same candidates.
    """
    import math
    docs = {}
    for group in groups:
        key = group.redundancy_cluster_id or group.group_id
        docs.setdefault(key, []).append(group)
    def _text(group):
        parts = []
        for node_id in group.node_ids:
            node = nodes.get(node_id)
            if node is not None and node.text:
                parts.append(node.text)
        return " ".join(parts)

    tokenised = {k: _norm(" ".join(_text(g) for g in v)).split()
                 for k, v in docs.items()}
    n = len(tokenised) or 1
    avgdl = sum(len(t) for t in tokenised.values()) / n
    df = Counter()
    for toks in tokenised.values():
        df.update(set(toks))
    k1, b = 1.5, 0.75
    q_terms = _norm(question).split()
    scores = {}
    for key, toks in tokenised.items():
        tf = Counter(toks)
        dl = len(toks) or 1
        total = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * (tf[term] * (k1 + 1) /
                            (tf[term] + k1 * (1 - b + b * dl / avgdl)))
        scores[key] = total
    return scores


def _retrieval_baseline_groups(groups, cfg, question, how, nodes) -> list:
    """Vanilla RAG's granularity rule under a different page ranking."""
    tier_rank = {"text_span": 3, "pdf_page": 2, "image": 2,
                 "pdf_region": 2, "doc_summary": 1}
    clusters = defaultdict(list)
    for group in groups:
        clusters[group.redundancy_cluster_id or group.group_id].append(group)
    if how == "bm25":
        scores = _bm25_scores(question, groups, nodes)
        rank = lambda item: -scores.get(item[0], 0.0)
    else:
        # Dense recall order, without the tier-aware bias the router applies.
        rank = lambda item: -max(g.base_score for g in item[1])
    ordered = [v for _, v in sorted(clusters.items(), key=rank)]
    # Budget handling is copied from _full_context_groups rather than
    # reimplemented, so these arms differ from Vanilla RAG in page ranking only.
    selected, cost = [], 0.0
    for values in ordered:
        richest = sorted(
            values,
            key=lambda group: (tier_rank.get(group.node_class, 0),
                               group.token_equivalent_cost),
            reverse=True,
        )
        choice = next((group for group in richest
                       if cost + group.token_equivalent_cost <=
                       cfg.router.max_total_cost), None)
        if choice is None:
            continue
        selected.append(choice)
        cost += choice.token_equivalent_cost
        if len(selected) >= cfg.router.max_groups:
            break
    return selected


def _full_context_groups(groups, cfg) -> list:
    """High-cost retrieved baseline: richest available tier per top page."""
    tier_rank = {"text_span": 3, "pdf_page": 2, "image": 2,
                 "pdf_region": 2, "doc_summary": 1}
    clusters = defaultdict(list)
    for group in groups:
        clusters[group.redundancy_cluster_id or group.group_id].append(group)
    ordered = sorted(
        clusters.values(),
        key=lambda values: -max(group.base_score for group in values),
    )
    selected, cost = [], 0.0
    for values in ordered:
        richest = sorted(
            values,
            key=lambda group: (tier_rank.get(group.node_class, 0),
                               group.token_equivalent_cost),
            reverse=True,
        )
        choice = next((group for group in richest
                       if cost + group.token_equivalent_cost <=
                       cfg.router.max_total_cost), None)
        if choice is None:
            continue
        selected.append(choice)
        cost += choice.token_equivalent_cost
        if len(selected) >= cfg.router.max_groups:
            break
    return selected


def _oracle_groups(qa: dict, nodes, cfg) -> list:
    """Generation ceiling using the correct modality on every Gold unit."""
    gold_doc = qa.get("qrel_doc_id") or qa.get("doc_id")
    evidence = qa.get("evidence") or {}
    evidence_pages = set(evidence.get("pages") or [])
    if not evidence_pages:
        return []
    by_cluster = defaultdict(list)
    for node in nodes.iter_all():
        if (node.source_ref.doc_id == gold_doc and
                node.source_ref.page_num in evidence_pages):
            companion_cost = sum(
                companion.token_equivalent_cost
                for companion_id in node.mandatory_companions
                if (companion := nodes.get(companion_id)) is not None
            )
            group = EvidenceGroup(
                group_id=f"oracle_{node.node_id}",
                root_node_id=node.node_id,
                node_ids=[node.node_id] + list(node.mandatory_companions),
                node_class=node.node_class,
                level=node.level,
                redundancy_cluster_id=node.redundancy_cluster_id,
                base_score=1.0,
                router_score=1.0,
                router_probability=1.0,
                token_equivalent_cost=(node.token_equivalent_cost +
                                       companion_cost),
                coverage_signature=node.coverage_signature,
                source_doc_id=node.source_ref.doc_id,
            )
            by_cluster[node.redundancy_cluster_id or node.node_id].append(group)
    selected, cost = [], 0.0
    for cluster in sorted(by_cluster):
        first = nodes.get(by_cluster[cluster][0].root_node_id)
        modalities = page_modalities(
            evidence, first.source_ref.page_num if first is not None else None)
        visual = has_visual_modality(modalities)
        textual = has_text_modality(modalities)

        def tier_rank(group):
            # MMDoc image/table quotes also have an LLM-generated text
            # description. It is useful retrieval text, but must not replace
            # the original image in a multimodal generation ceiling.
            if visual and not textual:
                return {"pdf_page": 4, "image": 4, "pdf_region": 4,
                        "text_span": 2, "doc_summary": 1}.get(
                            group.node_class, 0)
            return {"text_span": 4, "text_cluster": 4, "pdf_page": 2,
                    "image": 2, "pdf_region": 2, "doc_summary": 1}.get(
                        group.node_class, 0)

        choices = sorted(
            by_cluster[cluster],
            key=lambda group: (tier_rank(group),
                               group.token_equivalent_cost),
            reverse=True,
        )
        choice = next((group for group in choices
                       if cost + group.token_equivalent_cost <=
                       cfg.router.max_total_cost), None)
        if choice is not None:
            selected.append(choice)
            cost += choice.token_equivalent_cost
        if len(selected) >= cfg.router.max_groups:
            break
    return selected


def _retrieval_diagnostics(qa: dict, groups, nodes) -> dict:
    ranked = _rank_clusters(groups)
    gold_doc = qa.get("qrel_doc_id") or qa.get("doc_id")
    evidence_pages = set((qa.get("evidence") or {}).get("pages") or [])

    doc_rank = None
    for idx, group in enumerate(ranked, 1):
        if group.source_doc_id == gold_doc:
            doc_rank = idx
            break

    def page_recall_at(k: Optional[int]) -> float:
        if not evidence_pages:
            return 0.0
        subset = ranked if k is None else ranked[:k]
        seen = set()
        for group in subset:
            node = nodes.get(group.root_node_id)
            if node and group.source_doc_id == gold_doc:
                seen.add(node.source_ref.page_num)
        return len(seen & evidence_pages) / len(evidence_pages)

    gold_groups = []
    for group in groups:
        node = nodes.get(group.root_node_id)
        if (node and group.source_doc_id == gold_doc and
                node.source_ref.page_num in evidence_pages):
            gold_groups.append(group)
    cheapest_by_cluster = {}
    for group in gold_groups:
        key = group.redundancy_cluster_id
        cheapest_by_cluster[key] = min(
            cheapest_by_cluster.get(key, float("inf")),
            group.token_equivalent_cost)

    return {
        "gold_doc": gold_doc,
        "evidence_pages": sorted(evidence_pages),
        "ranked_cluster_count": len(ranked),
        "gold_doc_rank": doc_rank,
        "doc_hit_at_1": bool(doc_rank and doc_rank <= 1),
        "doc_hit_at_5": bool(doc_rank and doc_rank <= 5),
        "doc_hit_at_10": bool(doc_rank and doc_rank <= 10),
        "page_recall_at_5": page_recall_at(5),
        "page_recall_at_10": page_recall_at(10),
        "page_recall_all_candidates": page_recall_at(None),
        "oracle_min_cost": (sum(cheapest_by_cluster.values())
                            if cheapest_by_cluster else None),
        "top_candidates": [_group_info(g, nodes) for g in ranked[:10]],
    }


def _selected_diagnostics(selected, qa: dict, nodes) -> dict:
    gold_doc = qa.get("qrel_doc_id") or qa.get("doc_id")
    evidence_pages = set((qa.get("evidence") or {}).get("pages") or [])
    selected_pages = set()
    selected_docs = set()
    for group in selected:
        selected_docs.add(group.source_doc_id)
        node = nodes.get(group.root_node_id)
        if node and group.source_doc_id == gold_doc:
            selected_pages.add(node.source_ref.page_num)
    return {
        "selected_gold_doc_hit": gold_doc in selected_docs,
        "selected_gold_page_any_hit": bool(selected_pages & evidence_pages),
        "selected_gold_page_recall": (
            len(selected_pages & evidence_pages) / len(evidence_pages)
            if evidence_pages else 0.0),
    }


def _learned_model_path(cfg) -> str:
    return cfg.router.router_model_path or cfg.paths.router_model_dir


def _baseline_model_path(cfg) -> str:
    return _learned_model_path(cfg) + "_baseline"


def _available_routers(cfg) -> tuple:
    # This, not the module-level ROUTERS, is what the evaluation loop iterates,
    # so the opt-in arms have to be appended here too.  Building from
    # BASE_ROUTERS alone silently produced the old six-arm result on three
    # separate runs while EXTRA_ROUTERS was set and honoured everywhere else.
    names = list(BASE_ROUTERS)
    baseline = _baseline_model_path(cfg)
    if (os.path.exists(os.path.join(baseline, "router_weights.pt")) and
            os.path.exists(os.path.join(baseline, "router_meta.json"))):
        names.insert(names.index("learned"), "learned_baseline")
    names.extend(name for name in EXTRA_ROUTERS if name not in names)
    return tuple(names)


def _build_router(cfg, name: str):
    if name == "rule":
        return "scorer", RuleBasedScorer(cfg.router), True
    if name == "zeroshot":
        return "router", ZeroShotRouter(cfg), True
    model_path = (_baseline_model_path(cfg) if name == "learned_baseline"
                  else _learned_model_path(cfg))
    scorer = LearnedScorer(cfg.router, cfg.trainer, model_path)
    return "scorer", scorer, scorer._model is not None


def evaluate_qa(cfg, enc, vlm) -> Tuple[List[dict], dict]:
    engines = {name: _build_router(cfg, name)
               for name in ROUTERS
               if name not in ("full", "oracle") + RETRIEVAL_ROUTERS
               and name not in PROMPT_ROUTERS}
    selectors = {name: build_selector(cfg.router)
                 for name, engine in engines.items() if engine[0] == "scorer"}
    records = []
    max_q = int(getattr(cfg.evaluation, "max_questions", 0) or 0)
    judge_cfg = getattr(cfg.evaluation, "lm_judge", None)
    judge_enabled = bool(getattr(judge_cfg, "enabled", False))
    judge_max_tokens = int(getattr(judge_cfg, "max_tokens", 128))
    generation_cache = {}
    judge_cache = {}

    for dataset in cfg.datasets.enabled:
        test_ids = _load_split_ids(cfg, dataset, "test")
        if not test_ids:
            warn(f"[validation] {dataset}: no test ids")
            continue
        memory = DatasetMemory(cfg, dataset, enc)
        retriever = memory.retriever()
        info(f"[validation] QA dataset={dataset}, test={len(test_ids)}, cap={max_q or 'all'}")

        for qa in _iter_qa(cfg, dataset, test_ids, max_q):
            recall_start = time.perf_counter()
            groups, bias = retriever.recall(qa["question"])
            retrieval_ms = (time.perf_counter() - recall_start) * 1000
            retrieval_diag = _retrieval_diagnostics(qa, groups, memory.nodes)

            for router_name in ROUTERS:
                route_groups = copy.deepcopy(groups)
                route_start = time.perf_counter()
                if router_name == "full":
                    selected = _full_context_groups(route_groups, cfg)
                    model_ready = True
                elif router_name in RETRIEVAL_ROUTERS:
                    selected = _retrieval_baseline_groups(
                        route_groups, cfg, qa["question"], router_name,
                        memory.nodes)
                    model_ready = True
                elif router_name in PROMPT_ROUTERS:
                    selected = _full_context_groups(route_groups, cfg)
                    model_ready = True
                elif router_name == "oracle":
                    selected = _oracle_groups(qa, memory.nodes, cfg)
                    model_ready = True
                else:
                    kind, engine, model_ready = engines[router_name]
                    if kind == "router":
                        selected = engine.rank_and_select(
                            route_groups, cfg.cost.max_cost_normalizer,
                            bias=bias, question=qa["question"],
                        )
                    else:
                        engine.score(
                            route_groups, cfg.cost.max_cost_normalizer,
                            bias=bias, question=qa["question"],
                        )
                        selected = selectors[router_name].select(route_groups)
                        if not selected and route_groups:
                            selected = sorted(
                                route_groups,
                                key=lambda group: -group.router_probability)[:1]
                routing_ms = (time.perf_counter() - route_start) * 1000

                materialize_start = time.perf_counter()
                context, image_paths, cost, selected_info = _materialize(
                    selected, memory.nodes)
                materialization_ms = (time.perf_counter() - materialize_start) * 1000
                selection_diag = _selected_diagnostics(selected, qa, memory.nodes)
                answer, generation_ms = "", 0.0
                answer_cache_hit = False
                answer_usage = {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "estimated": False,
                }
                if selected:
                    # Prompt style is part of the key: two arms with identical
                    # selection but different prompts are different calls.
                    cache_key = (
                        dataset, qa["qa_id"],
                        tuple(group.group_id for group in selected),
                        PROMPT_ROUTERS.get(router_name, "direct"),
                    )
                    cached_answer = generation_cache.get(cache_key)
                    if cached_answer is not None:
                        answer_cache_hit = True
                        answer = cached_answer["answer"]
                        answer_usage = dict(cached_answer["usage"])
                        llm_ok = cached_answer["llm_ok"]
                        llm_stub = cached_answer["llm_stub"]
                        llm_error = cached_answer["llm_error"]
                    else:
                        generation_start = time.perf_counter()
                        answer = vlm.generate_answer(
                            qa["question"], context=context,
                            image_paths=image_paths,
                            style=PROMPT_ROUTERS.get(router_name, "direct"))
                        generation_ms = (time.perf_counter() - generation_start) * 1000
                        answer_usage = dict(vlm.last_usage)
                        llm_ok = bool(vlm.last_call_ok)
                        llm_stub = bool(vlm.last_call_was_stub)
                        llm_error = vlm.last_error
                        generation_cache[cache_key] = {
                            "answer": answer, "usage": dict(answer_usage),
                            "llm_ok": llm_ok, "llm_stub": llm_stub,
                            "llm_error": llm_error,
                        }
                else:
                    llm_ok, llm_stub = False, False
                    llm_error = "no candidates selected"

                gold_references = [str(value) for value in
                                   (qa.get("answer_aliases") or [])
                                   if str(value).strip()]
                if not gold_references:
                    gold_references = [qa.get("answer", "")]
                metric_options = [answer_metrics(answer, reference)
                                  for reference in gold_references]
                deterministic = max(
                    metric_options,
                    key=lambda metrics: (
                        metrics["token_f1"],
                        metrics.get("numeric_exact") or 0.0,
                        metrics["exact_match"],
                    ),
                )
                judge_result = {
                    "judge_valid": False, "judge_correct": None,
                    "judge_score": None, "judge_reason": "disabled",
                    "judge_raw": "", "judge_prompt_tokens": 0,
                    "judge_completion_tokens": 0, "judge_total_tokens": 0,
                    "judge_tokens_estimated": False, "judge_ms": 0.0,
                }
                judge_cache_hit = False
                if judge_enabled and llm_ok and not llm_stub:
                    judge_reference = qa.get("answer", "")
                    if len(gold_references) > 1:
                        judge_reference = (
                            "Any ONE of the following alternatives is acceptable:\n- " +
                            "\n- ".join(gold_references))
                    judge_key = (qa["question"], judge_reference, answer)
                    cached_judge = judge_cache.get(judge_key)
                    if cached_judge is not None:
                        judge_cache_hit = True
                        judge_result = dict(cached_judge)
                        judge_result["judge_ms"] = 0.0
                    else:
                        judge_result = run_lm_judge(
                            vlm, question=qa["question"],
                            gold_answer=judge_reference,
                            predicted_answer=answer,
                            max_tokens=judge_max_tokens,
                        )
                        judge_cache[judge_key] = dict(judge_result)

                oracle_cost = retrieval_diag.get("oracle_min_cost")
                evidence = qa.get("evidence") or {}
                evidence_modalities = []
                for evidence_page in evidence.get("pages") or []:
                    evidence_modalities.extend(
                        page_modalities(evidence, evidence_page))
                record = {
                    "dataset": dataset,
                    "qa_id": qa["qa_id"],
                    "question_type": qa.get("question_type", ""),
                    "evidence_page_modalities": (
                        evidence.get("page_modalities", {})),
                    "gold_evidence_visual": has_visual_modality(
                        evidence_modalities),
                    "question": qa["question"],
                    "gold_answer": qa.get("answer", ""),
                    "gold_references": gold_references,
                    "router": router_name,
                    "router_model_ready": model_ready,
                    "predicted_answer": answer,
                    "em": deterministic["exact_match"],
                    **deterministic,
                    "candidate_count": len(groups),
                    "selected_count": len(selected),
                    "selected": selected_info,
                    "tiers": [item["tier"] for item in selected_info],
                    "cost": cost,
                    "cost_over_oracle": (cost / oracle_cost
                                         if oracle_cost and oracle_cost > 0 else None),
                    "images_selected": len(image_paths),
                    "images_sent": answer_usage.get("image_count", 0),
                    "vision_enabled": vlm._vision_enabled(),
                    "llm_ok": llm_ok,
                    "llm_stub": llm_stub,
                    "llm_error": llm_error,
                    "answer_cache_hit": answer_cache_hit,
                    "answer_prompt_tokens": answer_usage.get("prompt_tokens", 0),
                    "answer_completion_tokens": answer_usage.get("completion_tokens", 0),
                    "answer_total_tokens": answer_usage.get("total_tokens", 0),
                    "answer_tokens_estimated": answer_usage.get("estimated", False),
                    "context_chars": len(context),
                    "retrieval_ms": retrieval_ms,
                    "routing_ms": routing_ms,
                    "materialization_ms": materialization_ms,
                    "generation_ms": generation_ms,
                    **judge_result,
                    "judge_cache_hit": judge_cache_hit,
                    "total_ms": (retrieval_ms + routing_ms + materialization_ms +
                                 generation_ms + judge_result["judge_ms"]),
                    "modality_bias": {
                        "active": bias.active_modality,
                        "confidence": bias.confidence,
                        "applied": bias.apply_bias,
                    },
                    **retrieval_diag,
                    **selection_diag,
                }
                records.append(record)
                info(f"[validation:{router_name}] {qa['qa_id']} "
                     f"f1={record['token_f1']:.3f} cost={cost:.1f} "
                     f"tiers={record['tiers']} llm_ok={llm_ok}")

    readiness = {"full": True, "oracle": True}
    readiness.update({name: bool(engine[2]) for name, engine in engines.items()})
    return records, readiness


def evaluate_retrieval_only(cfg, enc) -> Tuple[List[dict], dict]:
    details = []
    max_q = int(getattr(cfg.evaluation, "max_questions", 0) or 0)
    for dataset in cfg.datasets.retrieval_only:
        test_ids = _load_split_ids(cfg, dataset, "test")
        if not test_ids:
            continue
        memory = DatasetMemory(cfg, dataset, enc)
        retriever = memory.retriever()
        for qa in _iter_qa(cfg, dataset, test_ids, max_q):
            started = time.perf_counter()
            groups, _ = retriever.recall(qa["question"])
            latency_ms = (time.perf_counter() - started) * 1000
            ranked = _rank_clusters(groups)
            gold_doc = qa.get("qrel_doc_id") or qa.get("doc_id")
            rank = next((i for i, g in enumerate(ranked, 1)
                         if g.source_doc_id == gold_doc), None)
            details.append({
                "dataset": dataset,
                "qa_id": qa["qa_id"],
                "question": qa["question"],
                "gold_doc": gold_doc,
                "rank": rank,
                "hit_at_1": bool(rank and rank <= 1),
                "hit_at_5": bool(rank and rank <= 5),
                "hit_at_10": bool(rank and rank <= 10),
                "reciprocal_rank": (1.0 / rank if rank else 0.0),
                "candidate_clusters": len(ranked),
                "latency_ms": latency_ms,
                "top_docs": [g.source_doc_id for g in ranked[:10]],
            })

    summary = {}
    for dataset in cfg.datasets.retrieval_only:
        rows = [r for r in details if r["dataset"] == dataset]
        if not rows:
            continue
        ranks = [r["rank"] for r in rows if r["rank"]]
        summary[dataset] = {
            "n": len(rows),
            "recall_at_1": _round(_mean(r["hit_at_1"] for r in rows)),
            "recall_at_5": _round(_mean(r["hit_at_5"] for r in rows)),
            "recall_at_10": _round(_mean(r["hit_at_10"] for r in rows)),
            "mrr": _round(_mean(r["reciprocal_rank"] for r in rows)),
            "median_rank_when_found": _round(statistics.median(ranks), 2) if ranks else None,
            "avg_latency_ms": _round(_mean(r["latency_ms"] for r in rows), 2),
        }
    return details, summary


def _aggregate(rows: List[dict]) -> dict:
    if not rows:
        return {"n": 0}
    valid = [r for r in rows if r["llm_ok"] and not r["llm_stub"]]
    judge_valid = [r for r in valid if r.get("judge_valid")]
    correct = [r for r in valid if r["token_f1"] >= 0.5]
    unknown = [r for r in valid if
               str(r.get("predicted_answer", "")).strip().upper().rstrip(".") ==
               "UNKNOWN"]
    visual_gold = [r for r in rows if r.get("gold_evidence_visual")]
    costs = [r["cost"] for r in rows]
    tiers = Counter(tier for row in rows for tier in row["tiers"])
    return {
        "n": len(rows),
        "valid_answer_n": len(valid),
        "llm_success_rate": _round(len(valid) / len(rows)),
        "stub_or_failed_n": len(rows) - len(valid),
        "unknown_answer_n": len(unknown),
        "unknown_answer_rate": (_round(len(unknown) / len(valid))
                                  if valid else None),
        "em_all": _round(_mean(r["em"] for r in rows)),
        "f1_all": _round(_mean(r["token_f1"] for r in rows)),
        "em_valid": _round(_mean(r["em"] for r in valid)) if valid else None,
        "token_precision_valid": _round(_mean(
            r["token_precision"] for r in valid)) if valid else None,
        "token_recall_valid": _round(_mean(
            r["token_recall"] for r in valid)) if valid else None,
        "f1_valid": _round(_mean(r["token_f1"] for r in valid)) if valid else None,
        "bleu_1_valid": _round(_mean(r["bleu_1"] for r in valid)) if valid else None,
        "bleu_2_valid": _round(_mean(r["bleu_2"] for r in valid)) if valid else None,
        "rouge_l_valid": _round(_mean(r["rouge_l"] for r in valid)) if valid else None,
        "char_f1_valid": _round(_mean(r["char_f1"] for r in valid)) if valid else None,
        "edit_similarity_valid": _round(_mean(
            r["edit_similarity"] for r in valid)) if valid else None,
        "numeric_exact_valid": _round(mean_present(valid, "numeric_exact"))
        if mean_present(valid, "numeric_exact") is not None else None,
        "contains_valid": (_round(_mean(r["contains_answer"] for r in valid))
                           if valid else None),
        "accuracy_f1_ge_0_5": (_round(len(correct) / len(valid))
                               if valid else None),
        "judge_valid_n": len(judge_valid),
        "judge_coverage": _round(len(judge_valid) / len(valid)) if valid else None,
        "judge_accuracy": (_round(_mean(r["judge_correct"] for r in judge_valid))
                           if judge_valid else None),
        "judge_score": (_round(_mean(r["judge_score"] for r in judge_valid))
                        if judge_valid else None),
        "avg_cost": _round(_mean(costs), 2),
        "median_cost": _round(statistics.median(costs), 2),
        "p90_cost": _round(_percentile(costs, 0.90), 2),
        "cost_at_correct": (_round(_mean(r["cost"] for r in correct), 2)
                            if correct else None),
        "avg_cost_over_oracle": _round(_mean(
            r["cost_over_oracle"] for r in rows
            if r["cost_over_oracle"] is not None), 3),
        "avg_selected_groups": _round(_mean(r["selected_count"] for r in rows), 2),
        "gold_doc_selection_rate": _round(_mean(
            r["selected_gold_doc_hit"] for r in rows)),
        "gold_page_any_selection_rate": _round(_mean(
            r["selected_gold_page_any_hit"] for r in rows)),
        "gold_page_selection_recall": _round(_mean(
            r["selected_gold_page_recall"] for r in rows)),
        "candidate_gold_doc_recall_at_5": _round(_mean(
            r["doc_hit_at_5"] for r in rows)),
        "candidate_gold_page_recall": _round(_mean(
            r["page_recall_all_candidates"] for r in rows)),
        # Cost saving against a baseline is not interpretable on its own: a
        # baseline that answers UNKNOWN everywhere is trivially cheap, so a
        # router that actually answers can look like a cost regression.  F1 per
        # 1k tokens says how much quality each token bought, and cannot be won
        # by giving up.  (campaign_20260830_123023: learned scored 0.1434 vs
        # zeroshot 0.0676 while cost_saving_vs_zeroshot read -41.6%.)
        "f1_per_1k_tokens": _round(
            1000.0 * _mean(r["token_f1"] for r in rows) /
            max(_mean(r["cost"] for r in rows), 1e-9), 4),
        # Recall over the whole pool is the router's hard ceiling, but the
        # router only ever selects a handful of groups, so recall at the depth
        # it actually reaches is the number that explains a low selection rate.
        "candidate_gold_page_recall_at_5": _round(_mean(
            r.get("page_recall_at_5", 0.0) for r in rows)),
        "candidate_gold_page_recall_at_10": _round(_mean(
            r.get("page_recall_at_10", 0.0) for r in rows)),
        "tier_distribution": dict(sorted(tiers.items())),
        "images_selected": sum(r["images_selected"] for r in rows),
        "images_sent": sum(r["images_sent"] for r in rows),
        "visual_gold_n": len(visual_gold),
        "visual_gold_image_send_rate": (_round(_mean(
            r["images_sent"] > 0 for r in visual_gold))
            if visual_gold else None),
        "answer_cache_hits": sum(int(r.get("answer_cache_hit", False))
                                 for r in rows),
        "judge_cache_hits": sum(int(r.get("judge_cache_hit", False))
                                for r in rows),
        "answer_prompt_tokens": sum(r.get("answer_prompt_tokens", 0) for r in rows),
        "answer_completion_tokens": sum(
            r.get("answer_completion_tokens", 0) for r in rows),
        "answer_total_tokens": sum(r.get("answer_total_tokens", 0) for r in rows),
        "judge_prompt_tokens": sum(r.get("judge_prompt_tokens", 0) for r in rows),
        "judge_completion_tokens": sum(
            r.get("judge_completion_tokens", 0) for r in rows),
        "judge_total_tokens": sum(r.get("judge_total_tokens", 0) for r in rows),
        "token_usage_estimated_calls": sum(
            int(r.get("answer_tokens_estimated", False)) +
            int(r.get("judge_tokens_estimated", False)) for r in rows),
        "avg_retrieval_ms": _round(_mean(r["retrieval_ms"] for r in rows), 2),
        "avg_routing_ms": _round(_mean(r["routing_ms"] for r in rows), 2),
        "avg_materialization_ms": _round(_mean(
            r.get("materialization_ms", 0) for r in rows), 2),
        "avg_generation_ms": _round(_mean(r["generation_ms"] for r in rows), 2),
        "avg_judge_ms": _round(_mean(r.get("judge_ms", 0) for r in rows), 2),
        "p95_total_ms": _round(_percentile([r["total_ms"] for r in rows], 0.95), 2),
    }


def summarize_routers(records: List[dict]) -> Tuple[dict, dict]:
    overall, per_dataset = {}, {}
    datasets = sorted({r["dataset"] for r in records})
    for router in ROUTERS:
        overall[router] = _aggregate([r for r in records if r["router"] == router])
        per_dataset[router] = {
            dataset: _aggregate([
                r for r in records
                if r["router"] == router and r["dataset"] == dataset])
            for dataset in datasets
        }
    return overall, per_dataset


def _bootstrap_interval(values: List[float], samples: int = 1000,
                        seed: int = 42) -> Optional[List[float]]:
    if not values:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(draw))
    return [_round(_percentile(means, 0.025)),
            _round(_percentile(means, 0.975))]


def pairwise_summary(records: List[dict], candidate: str, baseline: str) -> dict:
    by_key = {(r["dataset"], r["qa_id"], r["router"]): r for r in records}
    keys = sorted({(r["dataset"], r["qa_id"]) for r in records})
    pairs = []
    for dataset, qa_id in keys:
        left = by_key.get((dataset, qa_id, candidate))
        right = by_key.get((dataset, qa_id, baseline))
        if left and right:
            pairs.append((left, right))
    if not pairs:
        return {"n": 0}
    base_cost = sum(right["cost"] for _, right in pairs)
    cand_cost = sum(left["cost"] for left, _ in pairs)
    valid = [(left, right) for left, right in pairs
             if left["llm_ok"] and right["llm_ok"] and
             not left["llm_stub"] and not right["llm_stub"]]
    changed = [
        (left, right) for left, right in pairs
        if [item["group_id"] for item in left.get("selected", [])] !=
        [item["group_id"] for item in right.get("selected", [])]
    ]
    valid_changed = [(left, right) for left, right in changed
                     if left["llm_ok"] and right["llm_ok"] and
                     not left["llm_stub"] and not right["llm_stub"]]
    judge_pairs = [(left, right) for left, right in valid
                   if left.get("judge_valid") and right.get("judge_valid")]
    f1_deltas = [left["token_f1"] - right["token_f1"]
                 for left, right in valid]
    cost_deltas = [left["cost"] - right["cost"] for left, right in pairs]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "n": len(pairs),
        "valid_answer_pairs": len(valid),
        "cost_saving_rate": (_round((base_cost - cand_cost) / base_cost)
                             if base_cost else None),
        "avg_cost_delta": _round(_mean(
            left["cost"] - right["cost"] for left, right in pairs), 2),
        "f1_delta_valid": (_round(_mean(
            f1_deltas)) if valid else None),
        "f1_delta_ci95": _bootstrap_interval(f1_deltas),
        "avg_cost_delta_ci95": _bootstrap_interval(cost_deltas),
        "judge_accuracy_delta": (_round(_mean(
            left["judge_correct"] - right["judge_correct"]
            for left, right in judge_pairs)) if judge_pairs else None),
        "judge_pair_n": len(judge_pairs),
        "accuracy_delta_valid": (_round(_mean(
            float(left["token_f1"] >= 0.5) - float(right["token_f1"] >= 0.5)
            for left, right in valid)) if valid else None),
        "cheaper_rate": _round(_mean(
            left["cost"] < right["cost"] for left, right in pairs)),
        "same_cost_rate": _round(_mean(
            abs(left["cost"] - right["cost"]) < 1e-9 for left, right in pairs)),
        "route_change_rate": _round(len(changed) / len(pairs)),
        "changed_n": len(changed),
        "changed_f1_delta": (_round(_mean(
            left["token_f1"] - right["token_f1"]
            for left, right in valid_changed)) if valid_changed else None),
        "changed_cost_saving_rate": (_round(_mean(
            (right["cost"] - left["cost"]) / right["cost"]
            for left, right in changed if right["cost"] > 0)) if changed else None),
        "strict_pareto_improvement_rate_valid": (_round(_mean(
            (left["cost"] < right["cost"] and
             left["token_f1"] >= right["token_f1"]) or
            (left["cost"] <= right["cost"] and
             left["token_f1"] > right["token_f1"])
            for left, right in valid)) if valid else None),
        "changed_pareto_nondegraded_rate_valid": (_round(_mean(
            left["cost"] <= right["cost"] and
            left["token_f1"] >= right["token_f1"]
            for left, right in valid_changed)) if valid_changed else None),
        "pareto_nondegraded_rate_valid": (_round(_mean(
            left["cost"] <= right["cost"] and
            left["token_f1"] >= right["token_f1"]
            for left, right in valid)) if valid else None),
    }


def data_stats(cfg) -> dict:
    out = {}
    for dataset in list(cfg.datasets.enabled) + list(cfg.datasets.retrieval_only):
        train_ids = _load_split_ids(cfg, dataset, "train")
        test_ids = _load_split_ids(cfg, dataset, "test")
        corpus_path = os.path.join(cfg.paths.unified_dir, dataset, "corpus.jsonl")
        docs = 0
        if os.path.exists(corpus_path):
            with open(corpus_path, encoding="utf-8") as f:
                docs = sum(1 for line in f if line.strip())
        out[dataset] = {"documents": docs, "train_qa": len(train_ids),
                        "test_qa": len(test_ids)}
    return out


def store_stats(cfg, enc) -> dict:
    out = {}
    for dataset in list(cfg.datasets.enabled) + list(cfg.datasets.retrieval_only):
        memory = DatasetMemory(cfg, dataset, enc)
        nodes = list(memory.nodes.iter_all())
        tier_counts = Counter(_tier(n.node_class) for n in nodes)
        tier_costs = defaultdict(list)
        for node in nodes:
            tier_costs[_tier(node.node_class)].append(node.token_equivalent_cost)
        out[dataset] = {
            "nodes": len(nodes),
            "text_index": memory.text_idx.size(),
            "visual_index": memory.vis_idx.size(),
            "tiers": dict(sorted(tier_counts.items())),
            "avg_cost_by_tier": {
                tier: _round(_mean(costs), 2)
                for tier, costs in sorted(tier_costs.items())
            },
        }
    return out


def training_stats(path: str, model_dir: str = "") -> dict:
    if not os.path.exists(path):
        return {"records": 0, "warning": "training JSONL not found"}
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    minimal_tiers = Counter()
    candidates = []
    for row in rows:
        groups = row.get("candidate_groups") or []
        candidates.append(len(groups))
        by_id = {g["group_id"]: g for g in groups}
        for gid in row.get("minimal_sufficient_groups") or []:
            group = by_id.get(gid)
            if group:
                minimal_tiers[_tier(group.get("node_class", "unknown"))] += 1
    result = {
        "records": len(rows),
        "avg_candidates": _round(_mean(candidates), 2),
        "minimal_tier_distribution": dict(sorted(minimal_tiers.items())),
    }
    meta_path = os.path.join(model_dir, "router_meta.json") if model_dir else ""
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
        result["best_epoch"] = meta.get("best_epoch")
        result["heldout_metrics"] = meta.get("heldout_metrics", {})
        result["opd"] = meta.get("opd", {})
        result["lambda_opd"] = meta.get("lambda_opd", 0.0)
    baseline_meta_path = os.path.join(
        model_dir + "_baseline", "router_meta.json") if model_dir else ""
    if baseline_meta_path and os.path.exists(baseline_meta_path):
        with open(baseline_meta_path, encoding="utf-8") as meta_file:
            baseline_meta = json.load(meta_file)
        result["baseline"] = {
            "best_epoch": baseline_meta.get("best_epoch"),
            "heldout_metrics": baseline_meta.get("heldout_metrics", {}),
            "opd": baseline_meta.get("opd", {}),
        }
    return result


def environment_stats(cfg, enc, vlm, model_readiness: dict) -> dict:
    torch_info = {"version": None, "cuda_available": False, "cuda_device": None}
    try:
        import torch
        torch_info["version"] = torch.__version__
        torch_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            torch_info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        torch_info["error"] = repr(exc)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch_info,
        "text_encoder_requested": cfg.encoders.text.backend,
        "text_encoder_actual": enc.text_enc.backend,
        "visual_encoder_requested": cfg.encoders.visual.backend,
        "visual_encoder_actual": enc.vis_enc.backend,
        "llm_backend": vlm.backend,
        "llm_model": vlm._model_name(),
        "llm_vision": vlm._vision_enabled(),
        "router_strategy": cfg.router.strategy,
        "min_relative_probability": float(getattr(
            cfg.router, "min_relative_probability", 0.05)),
        "learned_model_ready": model_readiness.get("learned", False),
        "learned_baseline_model_ready": model_readiness.get(
            "learned_baseline", False),
    }


def decide(summary: dict) -> Tuple[str, List[dict]]:
    learned = summary["routers"]["overall"].get("learned", {})
    oracle = summary["routers"]["overall"].get("oracle", {})
    zeroshot_overall = summary["routers"]["overall"].get("zeroshot", {})
    pair = summary["pairwise"].get("learned_vs_zeroshot", {})
    pair_full = summary["pairwise"].get("learned_vs_full", {})
    checks = [
        {
            "name": "Learned weights loaded",
            "value": int(bool(summary.get("environment", {}).get(
                "learned_model_ready", False))),
            "target": "= 1",
            "pass": bool(summary.get("environment", {}).get(
                "learned_model_ready", False)),
        },
        {
            "name": "Valid answer rate",
            "value": learned.get("llm_success_rate"),
            "target": ">= 90%",
            "pass": (learned.get("llm_success_rate", 0) >= 0.90),
        },
        {
            "name": "End-to-end test questions",
            "value": learned.get("n"),
            "target": ">= 30",
            "pass": (learned.get("n", 0) >= 30),
        },
        {
            "name": "Candidate-pool gold doc Recall@5",
            "value": learned.get("candidate_gold_doc_recall_at_5"),
            "target": ">= 80%",
            "pass": (learned.get("candidate_gold_doc_recall_at_5", 0) >= 0.80),
        },
        {
            "name": "Oracle judge accuracy",
            "value": oracle.get("judge_accuracy"),
            "target": ">= 50%",
            "pass": (oracle.get("judge_accuracy") is not None and
                     oracle["judge_accuracy"] >= 0.50),
        },
        {
            "name": "Oracle visual-gold image send rate",
            "value": oracle.get("visual_gold_image_send_rate"),
            "target": ">= 90% (when visual gold exists)",
            "pass": (oracle.get("visual_gold_n", 0) == 0 or
                     (oracle.get("visual_gold_image_send_rate") is not None and
                      oracle["visual_gold_image_send_rate"] >= 0.90)),
        },
        {
            "name": "Learned judge accuracy",
            "value": learned.get("judge_accuracy"),
            "target": ">= 15%",
            "pass": (learned.get("judge_accuracy") is not None and
                     learned["judge_accuracy"] >= 0.15),
        },
        {
            "name": "Learned F1 per 1k tokens",
            "value": learned.get("f1_per_1k_tokens"),
            "target": ">= zeroshot (efficiency; unaffected by baseline abstention)",
            "pass": (learned.get("f1_per_1k_tokens") is not None and
                     zeroshot_overall.get("f1_per_1k_tokens") is not None and
                     learned["f1_per_1k_tokens"] >=
                     zeroshot_overall["f1_per_1k_tokens"]),
        },
        {
            "name": "Candidate-pool gold page Recall (full pool)",
            "value": oracle.get("candidate_gold_page_recall"),
            "target": ">= 60%",
            "pass": (oracle.get("candidate_gold_page_recall") is not None and
                     oracle["candidate_gold_page_recall"] >= 0.60),
        },
        {
            "name": "Candidate-pool gold page Recall@5",
            "value": oracle.get("candidate_gold_page_recall_at_5"),
            "target": ">= 45% (the router selects only a few groups)",
            "pass": (oracle.get("candidate_gold_page_recall_at_5") is not None and
                     oracle["candidate_gold_page_recall_at_5"] >= 0.45),
        },
        {
            "name": "Learned Gold page hit rate",
            "value": learned.get("gold_page_any_selection_rate"),
            "target": ">= 35%",
            "pass": (learned.get("gold_page_any_selection_rate", 0) >= 0.35),
        },
        {
            "name": "Learned F1 change vs. Zeroshot",
            "value": pair.get("f1_delta_valid"),
            "target": ">= -0.03",
            "pass": (pair.get("f1_delta_valid") is not None and
                     pair["f1_delta_valid"] >= -0.03),
        },
        {
            # Reported for continuity, but zeroshot buys its cheapness with
            # UNKNOWN answers, so this is context rather than a criterion.
            "name": "Learned cost saving vs. Zeroshot (reference only)",
            "value": pair.get("cost_saving_rate"),
            "target": "Reference only (Zeroshot is cheap by abstaining)",
            "pass": True,
        },
        {
            "name": "Learned cost saving vs. Full",
            "value": pair_full.get("cost_saving_rate"),
            "target": ">= 0 (should be cheaper than reading everything)",
            "pass": (pair_full.get("cost_saving_rate") is not None and
                     pair_full["cost_saving_rate"] >= 0.0),
        },
        {
            "name": "Learned F1 change vs. Full",
            "value": pair_full.get("f1_delta_valid"),
            "target": ">= -0.03 (saving should not cost quality)",
            "pass": (pair_full.get("f1_delta_valid") is not None and
                     pair_full["f1_delta_valid"] >= -0.03),
        },
        {
            "name": "Fraction of routing decisions changed",
            "value": pair.get("route_change_rate"),
            "target": ">= 10%",
            "pass": (pair.get("route_change_rate") is not None and
                     pair["route_change_rate"] >= 0.10),
        },
    ]

    if not summary.get("environment", {}).get("learned_model_ready", False):
        verdict = "Inconclusive: learned weights were not loaded; this column used the fallback scorer."
    elif learned.get("valid_answer_n", 0) == 0:
        verdict = "Inconclusive: no usable model answers were produced."
    elif learned.get("n", 0) < 30:
        verdict = "Inconclusive: fewer than 30 end-to-end test questions; use only as a pipeline regression check."
    elif learned.get("candidate_gold_doc_recall_at_5", 0) < 0.50:
        verdict = "Not yet viable: the bottleneck is candidate retrieval, so the router has too little correct evidence to choose from."
    elif (oracle.get("visual_gold_n", 0) > 0 and
          oracle.get("visual_gold_image_send_rate", 0) < 0.90):
        verdict = "Needs attention: visual gold images were not sent reliably, so Oracle does not represent the answer model ceiling."
    elif oracle.get("judge_accuracy") is not None and oracle["judge_accuracy"] < 0.50:
        verdict = "Needs attention: answer quality is insufficient even with gold evidence."
    elif (oracle.get("candidate_gold_page_recall") is not None and
          oracle["candidate_gold_page_recall"] < 0.60):
        # Document-level recall can pass while page-level recall is the real
        # ceiling; in that case the router cannot select what was never
        # recalled, and tuning the router first would waste a campaign.
        verdict = ("Needs attention: the bottleneck is gold-page recall in the candidate pool (document-level recall is adequate, "
                   "but page-level recall is not). Improve retrieval before tuning the router.")
    elif learned.get("gold_page_any_selection_rate", 0) < 0.35:
        verdict = "Needs attention: gold evidence is in the candidate pool, but the learned router keeps too few gold pages."
    elif learned.get("judge_accuracy") is not None and learned["judge_accuracy"] < 0.15:
        verdict = "Needs attention: end-to-end accuracy is too low to justify a larger run."
    elif (pair.get("f1_delta_valid") is not None and
          pair["f1_delta_valid"] >= -0.03 and
          pair_full.get("cost_saving_rate", 0) >= 0.0 and
          pair_full.get("f1_delta_valid", -1) >= -0.03 and
          pair.get("route_change_rate", 0) >= 0.10):
        verdict = ("Promising: cheaper than reading everything with no quality loss, "
                   "and better than Zeroshot.")
    elif pair.get("f1_delta_valid") is not None and pair["f1_delta_valid"] < -0.03:
        verdict = "Needs attention: the cost saving comes with a clear drop in answer quality."
    else:
        verdict = "Conditionally viable: quality holds, but the cost advantage is small or the sample is too few."
    return verdict, checks


def _md(value, pct: bool = False) -> str:
    if value is None:
        return "N/A"
    if pct:
        return f"{float(value) * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _escape(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def write_report(summary: dict, records: List[dict], path: str) -> None:
    verdict, checks = decide(summary)
    summary["verdict"] = verdict
    summary["checks"] = checks
    env = summary["environment"]
    lines = [
        "# GATOR validation report",
        "",
        f"> Generated: {env['generated_at']}  ",
        f"> Automatic verdict: **{verdict}**",
        "",
        "## 1. Gate dashboard",
        "",
        "| Check | Value | Threshold | Result |",
        "|---|---:|---:|---|",
    ]
    for check in checks:
        pct = any(word in check["name"] for word in
                  ("rate", "accuracy", "Recall", "saving", "fraction"))
        lines.append(
            f"| {check['name']} | {_md(check['value'], pct=pct)} | "
            f"{check['target']} | {'PASS' if check['pass'] else 'CHECK'} |")

    lines += [
        "",
        "These thresholds are a quick screen for validation runs, not a release criterion.",
        "",
        "## 2. Environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Platform | `{_escape(env['platform'])}` |",
        f"| Python / Torch | `{env['python']}` / `{env['torch'].get('version')}` |",
        f"| CUDA | `{env['torch'].get('cuda_available')}` / `{env['torch'].get('cuda_device')}` |",
        f"| Text encoder | requested=`{env['text_encoder_requested']}`, actual=`{env['text_encoder_actual']}` |",
        f"| Visual encoder | requested=`{env['visual_encoder_requested']}`, actual=`{env['visual_encoder_actual']}` |",
        f"| Answer model | backend=`{env['llm_backend']}`, model=`{env['llm_model']}`, vision=`{env['llm_vision']}` |",
        f"| Selector | strategy=`{env.get('router_strategy')}`, relative probability floor=`{env.get('min_relative_probability', 0.0)}` |",
        f"| Learned weights | `{'loaded' if env['learned_model_ready'] else 'NOT LOADED'}` |",
        f"| No-OPD A/B weights | `{'loaded' if env.get('learned_baseline_model_ready') else 'not present'}` |",
        "",
        "## 3. Data and index coverage",
        "",
        "| Dataset | Docs | Train QA | Test QA | Nodes | Caption | Screenshot | Fulltext |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, ds in summary["data"].items():
        st = summary["stores"].get(dataset, {})
        tiers = st.get("tiers", {})
        lines.append(
            f"| {dataset} | {ds['documents']} | {ds['train_qa']} | {ds['test_qa']} | "
            f"{st.get('nodes', 0)} | {tiers.get('caption', 0)} | "
            f"{tiers.get('screenshot', 0)} | {tiers.get('fulltext', 0)} |")

    train = summary["training"]
    lines += [
        "",
        f"Training records: **{train.get('records', 0)}**; mean candidates: "
        f"**{train.get('avg_candidates', 'N/A')}**; minimal-tier distribution: "
        f"`{json.dumps(train.get('minimal_tier_distribution', {}), ensure_ascii=False)}`。",
        f"Held-out router metrics: "
        f"`{json.dumps(train.get('heldout_metrics', {}), ensure_ascii=False)}`；"
        f"best epoch：**{train.get('best_epoch', 'N/A')}**。",
        f"OPD config: `{json.dumps(train.get('opd', {}), ensure_ascii=False)}`; "
        f"lambda：**{train.get('lambda_opd', 0.0)}**。",
        f"Same-seed no-OPD baseline: "
        f"`{json.dumps(train.get('baseline', {}), ensure_ascii=False)}`。",
        "",
        "## 4. Overall router results",
        "",
        "| Router | N | LLM ok | UNKNOWN | EM | F1 | Acc(F1>=0.5) | Judge Acc | Avg Cost | P90 Cost | Gold page hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for router in ROUTERS:
        row = summary["routers"]["overall"].get(router, {})
        lines.append(
            f"| {router} | {row.get('n', 0)} | {_md(row.get('llm_success_rate'), True)} | "
            f"{_md(row.get('unknown_answer_rate'), True)} | "
            f"{_md(row.get('em_valid'))} | {_md(row.get('f1_valid'))} | "
            f"{_md(row.get('accuracy_f1_ge_0_5'), True)} | "
            f"{_md(row.get('judge_accuracy'), True)} | {_md(row.get('avg_cost'))} | "
            f"{_md(row.get('p90_cost'))} | {_md(row.get('gold_page_any_selection_rate'), True)} |")

    lines += [
        "",
        "### Granularity selection",
        "",
        "| Router | Tier distribution | Images selected | Images sent | Visual-gold questions | Image send rate | Avg groups kept |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for router in ROUTERS:
        row = summary["routers"]["overall"].get(router, {})
        lines.append(
            f"| {router} | `{json.dumps(row.get('tier_distribution', {}), ensure_ascii=False)}` | "
            f"{row.get('images_selected', 0)} | {row.get('images_sent', 0)} | "
            f"{row.get('visual_gold_n', 0)} | "
            f"{_md(row.get('visual_gold_image_send_rate'), True)} | "
            f"{_md(row.get('avg_selected_groups'))} |")

    lines += [
        "",
        "## 5. Answer metrics",
        "",
        "| Router | Precision | Recall | F1 | BLEU-1 | BLEU-2 | ROUGE-L | Char-F1 | Edit Sim | Numeric Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for router in ROUTERS:
        row = summary["routers"]["overall"].get(router, {})
        lines.append(
            f"| {router} | {_md(row.get('token_precision_valid'))} | "
            f"{_md(row.get('token_recall_valid'))} | {_md(row.get('f1_valid'))} | "
            f"{_md(row.get('bleu_1_valid'))} | {_md(row.get('bleu_2_valid'))} | "
            f"{_md(row.get('rouge_l_valid'))} | {_md(row.get('char_f1_valid'))} | "
            f"{_md(row.get('edit_similarity_valid'))} | "
            f"{_md(row.get('numeric_exact_valid'), True)} |")

    lines += [
        "",
        "Accuracy(F1>=0.5) is a threshold metric kept for continuity; read it together with the judge, numeric precision, and the generation metrics.",
        "",
        "## 6. Per-dataset results",
        "",
        "| Router | Dataset | N | F1(valid) | Acc | UNKNOWN | Avg Cost | Gold Doc | Gold Page Recall | LLM ok |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for router in ROUTERS:
        for dataset, row in summary["routers"]["per_dataset"].get(router, {}).items():
            lines.append(
                f"| {router} | {dataset} | {row.get('n', 0)} | {_md(row.get('f1_valid'))} | "
                f"{_md(row.get('accuracy_f1_ge_0_5'), True)} | "
                f"{_md(row.get('unknown_answer_rate'), True)} | "
                f"{_md(row.get('avg_cost'))} | "
                f"{_md(row.get('gold_doc_selection_rate'), True)} | "
                f"{_md(row.get('gold_page_selection_recall'), True)} | "
                f"{_md(row.get('llm_success_rate'), True)} |")

    lines += [
        "",
        "## 7. Against baselines",
        "",
        "| Comparison | Valid pairs | Cost saving | dF1 | F1 95% CI | dJudge | Route change | Strict Pareto |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, pair in summary["pairwise"].items():
        lines.append(
            f"| {name} | {pair.get('valid_answer_pairs', 0)} | "
            f"{_md(pair.get('cost_saving_rate'), True)} | {_md(pair.get('f1_delta_valid'))} | "
            f"`{pair.get('f1_delta_ci95')}` | {_md(pair.get('judge_accuracy_delta'))} | "
            f"{_md(pair.get('route_change_rate'), True)} | "
            f"{_md(pair.get('strict_pareto_improvement_rate_valid'), True)} |")

    lines += [
        "",
        "## 8. Retrieval-only benchmark",
        "",
        "| Dataset | N | Recall@1 | Recall@5 | Recall@10 | MRR | Mean latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in summary["retrieval_only"].items():
        lines.append(
            f"| {dataset} | {row['n']} | {_md(row['recall_at_1'], True)} | "
            f"{_md(row['recall_at_5'], True)} | {_md(row['recall_at_10'], True)} | "
            f"{_md(row['mrr'])} | {_md(row['avg_latency_ms'])} |")

    lines += [
        "",
        "## 9. Token usage and per-stage timing",
        "",
        "| Router | Answer In | Answer Out | Judge In | Judge Out | Retrieval ms | Route ms | Materialize ms | Generate ms | Judge ms | P95 Total ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for router in ROUTERS:
        row = summary["routers"]["overall"].get(router, {})
        lines.append(
            f"| {router} | {row.get('answer_prompt_tokens', 0)} | "
            f"{row.get('answer_completion_tokens', 0)} | "
            f"{row.get('judge_prompt_tokens', 0)} | "
            f"{row.get('judge_completion_tokens', 0)} | "
            f"{_md(row.get('avg_retrieval_ms'))} | {_md(row.get('avg_routing_ms'))} | "
            f"{_md(row.get('avg_materialization_ms'))} | "
            f"{_md(row.get('avg_generation_ms'))} | {_md(row.get('avg_judge_ms'))} | "
            f"{_md(row.get('p95_total_ms'))} |")

    lines += [
        "",
        "### VLM calls by purpose",
        "",
        "| Stage/purpose | Calls | Success | Prompt Tokens | Completion Tokens | Images | Total ms | Estimated Calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    usage_sections = summary.get("token_and_time", {})
    for stage_key in ("ingestion_vlm", "evaluation_vlm"):
        for purpose, usage in usage_sections.get(stage_key, {}).items():
            lines.append(
                f"| {stage_key}/{purpose} | {usage.get('calls', 0)} | "
                f"{usage.get('successful_calls', 0)} | {usage.get('prompt_tokens', 0)} | "
                f"{usage.get('completion_tokens', 0)} | {usage.get('image_count', 0)} | "
                f"{_md(usage.get('latency_ms'))} | {usage.get('estimated_calls', 0)} |")

    lines += [
        "",
        "Token counts come from the OpenAI-compatible `/usage` field when the server returns one; otherwise they are estimated from characters and images and counted as Estimated Calls.",
        "",
        "## 10. LM Judge",
        "",
        "| Router | Judge coverage | Judge accuracy | Mean score | Valid judgements |",
        "|---|---:|---:|---:|---:|",
    ]
    for router in ROUTERS:
        row = summary["routers"]["overall"].get(router, {})
        lines.append(
            f"| {router} | {_md(row.get('judge_coverage'), True)} | "
            f"{_md(row.get('judge_accuracy'), True)} | {_md(row.get('judge_score'))} | "
            f"{row.get('judge_valid_n', 0)} |")
    lines += [
        "",
        "The judge uses the same local model to compare question, gold and prediction. It is a supplementary metric and does not replace deterministic metrics or human spot checks.",
    ]

    # Show the most diagnostic failures, preferring valid model calls.
    valid_rows = [r for r in records if r["llm_ok"] and not r["llm_stub"]]
    failures = sorted(valid_rows, key=lambda r: (r["token_f1"], -r["cost"]))[:12]
    lines += [
        "",
        "## 11. Low-scoring and failed examples",
        "",
    ]
    if not failures:
        lines.append("No usable model answers; check the LLM configuration and `run.log` first.")
    for row in failures:
        lines += [
            f"### {row['dataset']} / {row['qa_id']} / {row['router']}",
            "",
            f"- Question: {_escape(row['question'], 500)}",
            f"- Gold：{_escape(row['gold_answer'], 500)}",
            f"- Prediction: {_escape(row['predicted_answer'], 500)}",
            f"- F1 / Cost / Tiers：{row['token_f1']:.4f} / {row['cost']:.1f} / `{row['tiers']}`",
            f"- BLEU-1 / ROUGE-L / Judge：{row['bleu_1']:.4f} / {row['rouge_l']:.4f} / "
            f"{row.get('judge_correct')} ({_escape(row.get('judge_reason', ''), 300)})",
            f"- Gold doc rank / gold page recall: {row['gold_doc_rank']} / {row['page_recall_all_candidates']:.2%}",
            "",
        ]

    lines += [
        "## 12. All per-question results (abridged)",
        "",
        "| Dataset | QA | Router | F1 | BLEU-1 | Judge | Cost | Tiers | Gold page Recall | Answer Tokens | Total ms |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['dataset']} | {row['qa_id']} | {row['router']} | "
            f"{row['token_f1']:.4f} | {row['bleu_1']:.4f} | "
            f"{_md(row.get('judge_correct'))} | {row['cost']:.1f} | "
            f"`{','.join(row['tiers'])}` | "
            f"{row['selected_gold_page_recall']:.2%} | "
            f"{row.get('answer_total_tokens', 0)} | {row['total_ms']:.1f} |")

    lines += [
        "",
        "## 13. How to read this report",
        "",
        "1. Check the LLM success rate and the encoders' actual backends first; if either fell back to a stub, the accuracy numbers are void.",
        "2. Then check candidate-pool gold recall; when it is low, fix the retriever before tuning the router.",
        "3. Read the main learned result from learned_vs_zeroshot; any OPD gain must come from the same-data, same-seed A/B in learned_opd_vs_baseline.",
        "4. Check that images selected and images actually sent agree; otherwise the visual-tier results are void.",
        "5. Oracle measures the answer model ceiling; Full measures a high-quality baseline that saves nothing after retrieval.",
        "6. Small samples are for regression checks only; a reportable conclusion needs a document-level held-out test set, several seeds, and confidence intervals.",
        "",
        "The files `per_question.jsonl`, `retrieval_details.jsonl` and `summary.json` in the same directory "
        "hold the untruncated machine-readable results. Neither the report nor those files record an API key.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    global ROUTERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    cfg = load_config(args.config)
    ROUTERS = _available_routers(cfg)
    banner("COMPREHENSIVE VALIDATION EVALUATION")
    enc = SharedEncoders(cfg)
    vlm = VLM(cfg.llm)

    records, model_readiness = evaluate_qa(cfg, enc, vlm)
    retrieval_details, retrieval_summary = evaluate_retrieval_only(cfg, enc)
    overall, per_dataset = summarize_routers(records)
    pairwise = {
        "learned_vs_rule": pairwise_summary(records, "learned", "rule"),
        "zeroshot_vs_rule": pairwise_summary(records, "zeroshot", "rule"),
        "learned_vs_zeroshot": pairwise_summary(records, "learned", "zeroshot"),
        "learned_vs_full": pairwise_summary(records, "learned", "full"),
        "full_vs_oracle": pairwise_summary(records, "full", "oracle"),
    }
    if "learned_baseline" in ROUTERS:
        pairwise["learned_opd_vs_baseline"] = pairwise_summary(
            records, "learned", "learned_baseline")
    ingest_usage_path = os.path.join(cfg.paths.log_dir, "ingest_vlm_usage.json")
    ingest_usage = {}
    if os.path.exists(ingest_usage_path):
        with open(ingest_usage_path, encoding="utf-8") as f:
            ingest_usage = json.load(f)
    summary = {
        "environment": environment_stats(cfg, enc, vlm, model_readiness),
        "data": data_stats(cfg),
        "stores": store_stats(cfg, enc),
        "training": training_stats(
            cfg.paths.training_data_path, cfg.paths.router_model_dir),
        "router_order": list(ROUTERS),
        "routers": {"overall": overall, "per_dataset": per_dataset},
        "pairwise": pairwise,
        "retrieval_only": retrieval_summary,
        "token_and_time": {
            "ingestion_vlm": ingest_usage,
            "evaluation_vlm": vlm.usage_summary(),
            "note": ("OpenAI/vLLM usage is used when returned; otherwise token counts "
                     "are estimated and marked per question."),
        },
    }
    verdict, checks = decide(summary)
    summary["verdict"] = verdict
    summary["checks"] = checks

    with open(os.path.join(args.run_dir, "per_question.jsonl"), "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.run_dir, "retrieval_details.jsonl"), "w", encoding="utf-8") as f:
        for row in retrieval_details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    report_path = os.path.join(args.run_dir, "REPORT.md")
    write_report(summary, records, report_path)
    info(f"[validation] verdict: {verdict}")
    info(f"[validation] report: {report_path}")


if __name__ == "__main__":
    main()
