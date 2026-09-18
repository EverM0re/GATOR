"""Stage 6: evaluate on the TEST split with three routers and compare.

For each router in evaluation.routers (rule | zeroshot | learned):
  - recall candidate groups for each test QA
  - score+select (cost-sensitive)
  - materialize context, generate answer (stub/LLM)
  - score EM / token-F1; record cost, selected tier distribution
Also computes Recall@k for retrieval-only datasets (vidore, real_mm_rag):
  - whether the gold page tier of the right doc is in the top-k recalled.

Writes:
  logs/eval_<router>.jsonl   per-question records
  logs/summary.json          final aggregated metrics (all routers)

    python -m scripts.evaluate --config config/file_router.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config                  # noqa: E402
from file_router.evaluation import answer_metrics, run_lm_judge  # noqa: E402
from file_router.memory import SharedEncoders, DatasetMemory  # noqa: E402
from file_router.router.scorer import RuleBasedScorer, LearnedScorer  # noqa: E402
from file_router.router.selector import build_selector       # noqa: E402
from file_router.router.zeroshot import ZeroShotRouter       # noqa: E402
from file_router.utils import banner, dbg, info, warn        # noqa: E402


# ----------------------------------------------------------------- metrics
def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred, gold):
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def token_f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def _tier_of(node_class):
    return {"doc_summary": "caption", "pdf_page": "screenshot",
            "text_span": "fulltext"}.get(node_class, node_class)


# ----------------------------------------------------------------- helpers
def _load_split_ids(cfg, ds, which):
    p = os.path.join(cfg.paths.splits_dir, f"{ds}.{which}.txt")
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return {l.strip() for l in f if l.strip()}


def _iter_qa(cfg, ds, keep_ids, limit=0):
    p = os.path.join(cfg.paths.unified_dir, ds, "qa.jsonl")
    if not os.path.exists(p):
        return
    n = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["qa_id"] in keep_ids:
                yield rec
                n += 1
                if limit and n >= limit:
                    return


def _make_scorer(cfg, name):
    if name == "rule":
        return ("scorer", RuleBasedScorer(cfg.router))
    if name == "learned":
        path = cfg.router.router_model_path or cfg.paths.router_model_dir
        return ("scorer", LearnedScorer(cfg.router, cfg.trainer, path))
    if name == "zeroshot":
        return ("router", ZeroShotRouter(cfg))
    raise ValueError(name)


def _materialize(groups, nodes):
    total = 0.0
    lines = []
    image_paths = []
    for g in groups:
        total += g.token_equivalent_cost
        for nid in g.node_ids:
            n = nodes.get(nid)
            if n is None:
                continue
            if n.text:
                lines.append(f"[{n.node_class} L{n.level} p{n.source_ref.page_num}] {n.text}")
            elif n.page_image_path:
                lines.append(f"[pdf_page p{n.source_ref.page_num} path={n.page_image_path}]")
                image_paths.append(n.page_image_path)
            elif n.image_path:
                lines.append(f"[image p{n.source_ref.page_num} path={n.image_path}]")
                image_paths.append(n.image_path)
    return "\n\n".join(lines), list(dict.fromkeys(image_paths)), total


# ----------------------------------------------------------------- QA eval
def eval_router_on_qa(cfg, enc, router_name, vlm):
    kind, obj = _make_scorer(cfg, router_name)
    selector = build_selector(cfg.router)
    recs = []
    for ds in cfg.datasets.enabled:
        test_ids = _load_split_ids(cfg, ds, "test")
        if not test_ids:
            continue
        mem = DatasetMemory(cfg, ds, enc)
        retr = mem.retriever()
        for qa in _iter_qa(cfg, ds, test_ids, cfg.evaluation.max_questions):
            groups, bias = retr.recall(qa["question"])
            if not groups:
                continue
            if kind == "router":          # zeroshot: scores+selects itself
                selected = obj.rank_and_select(
                    groups, cfg.cost.max_cost_normalizer,
                    bias=bias, question=qa["question"],
                )
            else:
                obj.score(
                    groups, cfg.cost.max_cost_normalizer,
                    bias=bias, question=qa["question"],
                )
                selected = selector.select(groups)
                if not selected:
                    selected = sorted(groups, key=lambda g: -g.router_probability)[:1]
            context, image_paths, cost = _materialize(selected, mem.nodes)
            answer = vlm.generate_answer(
                qa["question"], context=context, image_paths=image_paths)
            answer_usage = dict(vlm.last_usage)
            llm_ok = bool(vlm.last_call_ok and not vlm.last_call_was_stub)
            metrics = answer_metrics(answer, qa["answer"])
            judge = {"judge_valid": False, "judge_correct": None,
                     "judge_score": None, "judge_total_tokens": 0,
                     "judge_ms": 0.0}
            judge_cfg = getattr(cfg.evaluation, "lm_judge", None)
            if llm_ok and bool(getattr(judge_cfg, "enabled", False)):
                judge = run_lm_judge(
                    vlm, qa["question"], qa["answer"], answer,
                    max_tokens=int(getattr(judge_cfg, "max_tokens", 128)),
                )
            tiers = [_tier_of(g.node_class) for g in selected]
            rec = {"qa_id": qa["qa_id"], "dataset": ds,
                   "em": metrics["exact_match"], "f1": metrics["token_f1"],
                   **metrics, **judge,
                   "cost": cost, "n_selected": len(selected), "tiers": tiers,
                   "answer": answer[:200], "images_attached": len(image_paths),
                   "answer_prompt_tokens": answer_usage.get("prompt_tokens", 0),
                   "answer_completion_tokens": answer_usage.get("completion_tokens", 0),
                   "answer_total_tokens": answer_usage.get("total_tokens", 0),
                   "llm_ok": llm_ok,
                   "llm_error": "" if llm_ok else vlm.last_error}
            recs.append(rec)
            dbg(f"[eval:{router_name}] {qa['qa_id']} "
                f"f1={metrics['token_f1']:.2f} cost={cost:.0f} tiers={tiers}")
    return recs


def summarize(recs):
    if not recs:
        return {"n": 0}
    n = len(recs)
    em = sum(r["em"] for r in recs) / n
    f1 = sum(r["f1"] for r in recs) / n
    cost = sum(r["cost"] for r in recs) / n
    correct = [r for r in recs if r["f1"] >= 0.5]
    cost_at_correct = (sum(r["cost"] for r in correct) / len(correct)) if correct else 0.0
    tier_counts = {}
    for r in recs:
        for t in r["tiers"]:
            tier_counts[t] = tier_counts.get(t, 0) + 1
    judge_rows = [r for r in recs if r.get("judge_valid")]
    return {"n": n, "avg_em": round(em, 4), "avg_f1": round(f1, 4),
            "avg_token_precision": round(sum(r["token_precision"] for r in recs) / n, 4),
            "avg_token_recall": round(sum(r["token_recall"] for r in recs) / n, 4),
            "avg_bleu_1": round(sum(r["bleu_1"] for r in recs) / n, 4),
            "avg_bleu_2": round(sum(r["bleu_2"] for r in recs) / n, 4),
            "avg_rouge_l": round(sum(r["rouge_l"] for r in recs) / n, 4),
            "judge_accuracy": (round(sum(r["judge_correct"] for r in judge_rows) /
                                     len(judge_rows), 4) if judge_rows else None),
            "answer_total_tokens": sum(r.get("answer_total_tokens", 0) for r in recs),
            "judge_total_tokens": sum(r.get("judge_total_tokens", 0) for r in recs),
            "avg_cost": round(cost, 1), "cost_at_correct": round(cost_at_correct, 1),
            "n_correct_f1>=0.5": len(correct), "tier_distribution": tier_counts}


# ----------------------------------------------------------- retrieval eval
def eval_retrieval(cfg, enc, k=5):
    """Recall@k: is the gold doc's any-tier node in the top-k text+visual recall?"""
    out = {}
    for ds in cfg.datasets.retrieval_only:
        test_ids = _load_split_ids(cfg, ds, "test")
        if not test_ids:
            continue
        mem = DatasetMemory(cfg, ds, enc)
        retr = mem.retriever()
        hit, tot = 0, 0
        for qa in _iter_qa(cfg, ds, test_ids, cfg.evaluation.max_questions):
            groups, _ = retr.recall(qa["question"])
            gold_doc = qa.get("qrel_doc_id") or qa["doc_id"]
            topk = sorted(groups, key=lambda g: -g.base_score)[:k]
            tot += 1
            if any(g.source_doc_id == gold_doc for g in topk):
                hit += 1
        out[ds] = {"n": tot, f"recall@{k}": round(hit / tot, 4) if tot else 0.0}
        info(f"[eval:retrieval] {ds} recall@{k}={out[ds][f'recall@{k}']} (n={tot})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/file_router.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    banner("EVALUATE on TEST split")
    enc = SharedEncoders(cfg)
    from file_router.encoders import VLM
    vlm = VLM(cfg.llm)

    summary = {"routers": {}, "retrieval": {}}
    for rname in cfg.evaluation.routers:
        info(f"--- evaluating router: {rname} ---")
        try:
            recs = eval_router_on_qa(cfg, enc, rname, vlm)
        except Exception as e:  # noqa: BLE001
            warn(f"[eval:{rname}] failed: {repr(e)[:160]}")
            continue
        with open(os.path.join(cfg.paths.log_dir, f"eval_{rname}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["routers"][rname] = summarize(recs)
        info(f"[eval:{rname}] {summary['routers'][rname]}")

    summary["retrieval"] = eval_retrieval(cfg, enc, k=5)
    summary["notes"] = {
        "llm_backend": cfg.llm.backend,
        "answer_metrics_meaningful": cfg.llm.backend != "stub",
        "hint": ("EM/F1 are ~0 under llm.backend=stub (placeholder answers). "
                 "Routing/cost/tier_distribution/retrieval metrics ARE real. "
                 "Set llm.backend to anthropic/openai for real answer correctness."),
    }

    with open(os.path.join(cfg.paths.log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    banner("EVAL SUMMARY")
    info(json.dumps(summary, indent=2, ensure_ascii=False))
    info(f"summary written -> {os.path.join(cfg.paths.log_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
