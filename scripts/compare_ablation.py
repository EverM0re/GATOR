"""Compare tier-ablation arms and say what each cost tier is actually worth.

The decision this supports: caption is priced at roughly a tenth of a page
screenshot, but in campaign_20260830_123023 a caption chosen on the Gold page
yielded 0.226 less F1 than oracle while a screenshot matched oracle exactly.
If removing a tier does not lower F1-per-token, that tier is not paying for the
place it occupies in the cost ladder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_metrics(arm_dir: Path) -> dict[str, Any] | None:
    runs = sorted(arm_dir.glob("validation_*/summary.json"))
    if not runs:
        return None
    summary = _read_json(runs[-1])
    overall = summary.get("routers", {}).get("overall", {})
    learned = overall.get("learned", {})
    zeroshot = overall.get("zeroshot", {})
    if not learned:
        return None
    return {
        "arm": arm_dir.name,
        "run_dir": str(runs[-1].parent),
        "f1": learned.get("f1_valid"),
        "judge": learned.get("judge_accuracy"),
        "cost": learned.get("avg_cost"),
        "f1_per_1k": learned.get("f1_per_1k_tokens"),
        "gold_page": learned.get("gold_page_any_selection_rate"),
        "unknown": learned.get("unknown_answer_rate"),
        "tier_distribution": learned.get("tier_distribution"),
        "zeroshot_f1_per_1k": zeroshot.get("f1_per_1k_tokens"),
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(value, base) -> str:
    if value is None or base is None:
        return "N/A"
    return f"{value - base:+.4f}"


TIER_ORDER = ("caption", "fulltext", "screenshot")


def _tier_on_gold_page(row: dict[str, Any]) -> str | None:
    """Which cost tier did this run actually read on the Gold page?

    Returns None when the Gold page was not selected at all.  Priority follows
    the cost ladder: the most informative tier present is the one the answer
    model could have used.
    """
    if not row.get("selected_gold_page_any_hit"):
        return None
    gold_doc = row.get("gold_doc")
    pages = set(row.get("evidence_pages") or [])
    tiers = {s.get("tier") for s in row.get("selected") or []
             if s.get("doc_id") == gold_doc and s.get("page") in pages}
    for tier in ("screenshot", "fulltext", "caption"):
        if tier in tiers:
            return tier
    return None


def pooled_tier_quality(root: Path) -> dict[str, dict[str, Any]]:
    """Answer quality grouped by the tier read on the Gold page, pooled over arms.

    Conditioning on a Gold-page hit holds page coverage constant, so this
    isolates what each tier is worth -- which the arm-level table cannot do once
    ablation changes how many pages fit in the budget.
    """
    buckets: dict[str, list[dict[str, Any]]] = {t: [] for t in TIER_ORDER}
    for per_question in sorted(root.glob("*/validation_*/per_question.jsonl")):
        with open(per_question, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("router") != "learned":
                    continue
                tier = _tier_on_gold_page(row)
                if tier in buckets:
                    buckets[tier].append(row)

    out: dict[str, dict[str, Any]] = {}
    for tier, rows in buckets.items():
        if not rows:
            continue
        f1 = sum(r.get("token_f1", 0.0) for r in rows) / len(rows)
        cost = sum(r.get("cost", 0.0) for r in rows) / len(rows)
        unknown = sum(
            1 for r in rows
            if str(r.get("predicted_answer", "")).strip().upper().startswith(
                "UNKNOWN")) / len(rows)
        out[tier] = {
            "n": len(rows), "f1": f1, "cost": cost, "unknown": unknown,
            "f1_per_1k": 1000.0 * f1 / cost if cost else None,
        }
    return out


def compare(root: Path) -> list[dict[str, Any]]:
    arms = [a for a in (_arm_metrics(d) for d in sorted(root.iterdir())
                        if d.is_dir()) if a]
    if not arms:
        raise SystemExit(f"no completed ablation arms under {root}")

    baseline = next((a for a in arms if a["arm"] == "baseline"), arms[0])
    lines = [
        "# Tier ablation (Experiment 1)", "",
        "Question: is each rung of the cost ladder worth the slot it occupies?", "",
        "How to read: **F1/1k tokens is the primary criterion**. If it rises when a rung is removed,",
        "that rung is a net liability at current quality: cheap but uninformative,",
        "consuming a selection slot that a costlier but effective rung should have had.", "",
        "| Arm | F1 | dF1 | Judge | UNKNOWN | Avg Cost | **F1/1k tokens** | d-efficiency | Gold page hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        marker = " (reference)" if arm is baseline else ""
        lines.append(
            f"| `{arm['arm']}`{marker} | {_fmt(arm['f1'])} | "
            f"{_delta(arm['f1'], baseline['f1'])} | {_fmt(arm['judge'])} | "
            f"{_fmt(arm['unknown'])} | "
            f"{_fmt(arm['cost'], 1)} | **{_fmt(arm['f1_per_1k'])}** | "
            f"{_delta(arm['f1_per_1k'], baseline['f1_per_1k'])} | "
            f"{_fmt(arm['gold_page'])} |")

    lines += ["", "## Tier distribution per arm", "",
              "A disabled tier can still appear: it is kept when a page has no other candidate,",
              "so that the question does not become evidence-free and contaminate the comparison.", "",
              "| Arm | Tier distribution |", "|---|---|"]
    for arm in arms:
        lines.append(f"| `{arm['arm']}` | `{json.dumps(arm['tier_distribution'], ensure_ascii=False)}` |")

    tier_quality = pooled_tier_quality(root)
    if tier_quality:
        lines += [
            "", "## Grouped by the tier actually read on gold pages (pooled over arms)", "",
            "Only questions that hit a gold page are counted, so page coverage is fixed; this",
            "is the section that isolates the value of a tier itself.", "",
            "| Tier | n | F1 | Avg Cost | F1/1k tokens | UNKNOWN |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for tier in TIER_ORDER:
            stat = tier_quality.get(tier)
            if not stat:
                continue
            lines.append(
                f"| {tier} | {stat['n']} | {_fmt(stat['f1'])} | "
                f"{_fmt(stat['cost'], 1)} | {_fmt(stat['f1_per_1k'])} | "
                f"{_fmt(stat['unknown'])} |")
        lines.append("")

    # Confound check: ablating a cheap tier forces pricier picks, which can hit
    # the cost budget and reduce how many PAGES get selected.  When that happens
    # the arms differ in page coverage as well as tier mix, and the headline
    # F1/1k comparison is not attributable to tier quality alone.
    lines += ["", "## Confound check", ""]
    base_pages = baseline.get("gold_page")
    confounded = [
        a for a in arms if a is not baseline and
        a.get("gold_page") is not None and base_pages is not None and
        base_pages - a["gold_page"] > 0.05
    ]
    if confounded:
        lines += [
            "**The arms below have a markedly lower gold-page hit rate than the reference**: disabling a cheap tier",
            "forces costlier picks, exhausts the budget sooner, and therefore selects **fewer pages**:", "",
            "| Arm | Gold page hit | vs. reference |", "|---|---:|---:|",
        ]
        for arm in confounded:
            lines.append(
                f"| `{arm['arm']}` | {_fmt(arm['gold_page'])} | "
                f"{_delta(arm['gold_page'], base_pages)} |")
        lines += [
            "",
            "These arms vary both tier and page coverage, so the F1/1k differences above",
            "**cannot** be attributed to tier quality alone. For that, see the section below,",
            "grouped by the tier read on gold pages, where page coverage is held fixed.",
            "",
        ]
    else:
        lines += ["Gold-page hit rates are comparable across arms, so page coverage is not a confound.", ""]

    # Verdict
    lines += ["## Reading the results", ""]
    best = max(arms, key=lambda a: a["f1_per_1k"] or 0)
    if confounded and best is baseline:
        lines.append(
            "The reference arm has the highest F1/1k, but the gap comes **at least partly from reduced page coverage rather than tier quality** "
            "(see the confound check above). This does not establish that the ladder is sound; read the per-tier "
            "grouping first, and check whether a dataset is missing its middle rung.")
    elif best is baseline:
        lines.append(
            "**The full cost ladder has the highest token efficiency**: every rung contributes, "
            "so the bottleneck is not the ladder design but retrieval ranking.")
    else:
        removed = best["arm"].replace("no_", "").replace("_", " + ")
        lines.append(
            f"**Removing `{removed}` raises token efficiency** "
            f"({_fmt(best['f1_per_1k'])} vs. reference {_fmt(baseline['f1_per_1k'])}). "
            f"That rung is a net liability at current quality: cheap, but uninformative, "
            f"while still consuming a selection slot. The fix is to improve its content (for example, generating "
            f"captions with a VLM instead of truncating heuristically), not to tune router parameters.")
    if baseline.get("zeroshot_f1_per_1k") is not None:
        lines += ["", f"Reference: zeroshot F1/1k tokens = "
                      f"{_fmt(baseline['zeroshot_f1_per_1k'])}。"]

    (root / "ABLATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    (root / "ablation_summary.json").write_text(
        json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")
    return arms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", required=True)
    args = parser.parse_args()
    root = Path(args.ablation_root)
    if not root.exists():
        parser.error(f"ablation root does not exist: {root}")
    arms = compare(root)
    print(f"[ABLATION] arms={len(arms)}")
    for arm in arms:
        print(f"[ABLATION] {arm['arm']}: F1={_fmt(arm['f1'])} "
              f"cost={_fmt(arm['cost'], 1)} "
              f"F1/1k={_fmt(arm['f1_per_1k'])}")
    print(f"[ABLATION] report={root / 'ABLATION_REPORT.md'}")


if __name__ == "__main__":
    main()
