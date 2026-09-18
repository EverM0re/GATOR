"""Stop expensive multi-seed runs when medium validation is not yet useful."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_gate(root: Path, subset: int, min_oracle_judge: float = 0.50,
               min_llm_success: float = 0.90,
               min_candidate_doc_recall: float = 0.80,
               min_learned_cost_saving: float = 0.0,
               min_oracle_visual_send: float = 0.90,
               min_learned_judge: float = 0.15,
               min_learned_gold_page_hit: float = 0.35,
               min_learned_f1_delta: float = 0.0,
               min_candidate_page_recall: float = 0.60,
               min_efficiency_ratio: float = 1.0,
               min_f1_delta_vs_full: float = -0.03) -> dict:
    candidates = []
    for manifest_path in (root / "runs").glob("validation_*/manifest.json"):
        manifest = _read(manifest_path)
        if (manifest.get("status") == "complete" and
                int(manifest.get("arguments", {}).get("subset", -1)) == subset and
                (manifest_path.parent / "summary.json").exists()):
            candidates.append(manifest_path)
    candidates.sort(key=lambda path: path.parent.name)
    if not candidates:
        return {"passed": False, "reason": f"no completed subset={subset} run"}

    manifest_path = candidates[-1]
    summary = _read(manifest_path.parent / "summary.json")
    oracle = summary.get("routers", {}).get("overall", {}).get("oracle", {})
    learned = summary.get("routers", {}).get("overall", {}).get("learned", {})
    zeroshot = summary.get("routers", {}).get("overall", {}).get("zeroshot", {})
    full_router = summary.get("routers", {}).get("overall", {}).get("full", {})
    pair_full = summary.get("pairwise", {}).get("learned_vs_full", {})

    # Cost saving against zeroshot is not a sound gate on its own: zeroshot gets
    # cheap by answering UNKNOWN.  In campaign_20260831_052759 it answered 85% of
    # questions with UNKNOWN at F1 0.0251, so a router that actually answers
    # necessarily "regresses" on that ratio.  Efficiency (F1 per token) cannot be
    # won by giving up, and the comparison against `full` checks the real claim:
    # cheaper than reading everything, without losing quality.
    learned_eff = learned.get("f1_per_1k_tokens")
    zeroshot_eff = zeroshot.get("f1_per_1k_tokens")
    efficiency_ratio = (
        learned_eff / zeroshot_eff
        if learned_eff is not None and zeroshot_eff else None)
    pair = summary.get("pairwise", {}).get("learned_vs_zeroshot", {})
    values = {
        "oracle_judge_accuracy": oracle.get("judge_accuracy"),
        "oracle_llm_success_rate": oracle.get("llm_success_rate"),
        "candidate_gold_doc_recall_at_5": oracle.get("candidate_gold_doc_recall_at_5"),
        # Document-level recall passed at 0.90 in campaign_20260823_010042
        # while page-level recall sat at 0.72 and page_recall@5 at 0.40 — the
        # actual ceiling on how often the router can select a Gold page.
        # Without this check the gate blames the router for a retrieval fault.
        "candidate_gold_page_recall": oracle.get("candidate_gold_page_recall"),
        "learned_cost_saving_vs_zeroshot": pair.get("cost_saving_rate"),
        "learned_efficiency_ratio_vs_zeroshot": (
            round(efficiency_ratio, 4) if efficiency_ratio is not None else None),
        "learned_cost_saving_vs_full": pair_full.get("cost_saving_rate"),
        "learned_f1_delta_vs_full": pair_full.get("f1_delta_valid"),
        "oracle_visual_gold_image_send_rate": oracle.get(
            "visual_gold_image_send_rate"),
        "learned_judge_accuracy": learned.get("judge_accuracy"),
        "learned_gold_page_selection_rate": learned.get(
            "gold_page_any_selection_rate"),
        "learned_f1_delta_vs_zeroshot": pair.get("f1_delta_valid"),
    }
    thresholds = {
        "oracle_judge_accuracy": min_oracle_judge,
        "oracle_llm_success_rate": min_llm_success,
        "candidate_gold_doc_recall_at_5": min_candidate_doc_recall,
        "candidate_gold_page_recall": min_candidate_page_recall,
        "learned_efficiency_ratio_vs_zeroshot": min_efficiency_ratio,
        "learned_cost_saving_vs_full": min_learned_cost_saving,
        "learned_f1_delta_vs_full": min_f1_delta_vs_full,
        "oracle_visual_gold_image_send_rate": min_oracle_visual_send,
        "learned_judge_accuracy": min_learned_judge,
        "learned_gold_page_selection_rate": min_learned_gold_page_hit,
        "learned_f1_delta_vs_zeroshot": min_learned_f1_delta,
    }
    # Reported for context but deliberately not gated: see the note above about
    # zeroshot buying cheapness with UNKNOWN answers.
    checks = {
        name: value is not None and float(value) >= thresholds[name]
        for name, value in values.items()
        if name in thresholds
    }
    # Text-only datasets do not make this check applicable.
    if int(oracle.get("visual_gold_n", 0) or 0) == 0:
        checks["oracle_visual_gold_image_send_rate"] = True
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(manifest_path.parent),
        "passed": all(checks.values()),
        "values": values,
        "thresholds": thresholds,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--subset", type=int, default=300)
    parser.add_argument("--min-oracle-judge", type=float, default=0.50)
    parser.add_argument("--min-llm-success", type=float, default=0.90)
    parser.add_argument("--min-candidate-doc-recall", type=float, default=0.80)
    parser.add_argument("--min-learned-cost-saving", type=float, default=0.0)
    parser.add_argument("--min-oracle-visual-send", type=float, default=0.90)
    parser.add_argument("--min-learned-judge", type=float, default=0.15)
    parser.add_argument("--min-learned-gold-page-hit", type=float, default=0.35)
    parser.add_argument("--min-learned-f1-delta", type=float, default=0.0)
    parser.add_argument("--min-candidate-page-recall", type=float, default=0.60)
    # Efficiency must at least match zeroshot; cost must beat reading everything.
    parser.add_argument("--min-efficiency-ratio", type=float, default=1.0)
    parser.add_argument("--min-f1-delta-vs-full", type=float, default=-0.03)
    args = parser.parse_args()
    root = Path(args.campaign_root)
    result = check_gate(
        root, args.subset, args.min_oracle_judge, args.min_llm_success,
        args.min_candidate_doc_recall, args.min_learned_cost_saving,
        args.min_oracle_visual_send,
        args.min_learned_judge, args.min_learned_gold_page_hit,
        args.min_learned_f1_delta, args.min_candidate_page_recall,
        args.min_efficiency_ratio, args.min_f1_delta_vs_full,
    )
    (root / "quality_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed") else 3


if __name__ == "__main__":
    raise SystemExit(main())
