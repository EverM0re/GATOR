"""Collect every validation result in a campaign into one compact handoff."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


AGGREGATED_METRICS = (
    ("learned_f1", "Learned F1"),
    ("learned_judge_accuracy", "Judge Acc"),
    ("learned_gold_page_selection_rate", "Gold page hit"),
    ("learned_avg_cost", "Avg cost"),
    ("cost_saving_rate", "vs Zeroshot saving"),
    ("f1_delta_valid", "vs Zeroshot F1 delta"),
)


def _metric_value(row: dict[str, Any], key: str):
    if key in ("cost_saving_rate", "f1_delta_valid"):
        return (row.get("learned_vs_zeroshot") or {}).get(key)
    return row.get(key)


def aggregate_by_scale(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean / std across seeds, grouped by the scale that produced them.

    A multi-seed large run is only meaningful as a distribution: a single seed
    cannot show whether an improvement survives resampling.  Runs are grouped by
    (subset, visual backend) so a CLIP medium and a ColPali medium in the same
    campaign are never averaged together.
    """
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subset = (row.get("scale_arguments") or {}).get("subset")
        groups[(subset, row.get("visual_backend"))].append(row)

    aggregates = []
    for (subset, visual), group in sorted(
            groups.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][1]))):
        entry: dict[str, Any] = {
            "subset": subset,
            "visual_backend": visual,
            "runs": len(group),
            "seeds": [row.get("split_seed") for row in group],
            "metrics": {},
        }
        for key, _label in AGGREGATED_METRICS:
            values = [v for v in (_metric_value(r, key) for r in group)
                      if isinstance(v, (int, float))]
            if not values:
                continue
            entry["metrics"][key] = {
                "mean": round(statistics.mean(values), 4),
                # Population of one has no spread; report 0.0 rather than error.
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": len(values),
            }
        aggregates.append(entry)
    return aggregates


def summarize(campaign_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = campaign_root / "runs"
    for summary_path in sorted(pattern.glob("validation_*/summary.json")):
        summary = _read_json(summary_path)
        manifest = _read_json(summary_path.parent / "manifest.json")
        learned = summary.get("routers", {}).get("overall", {}).get("learned", {})
        pair = summary.get("pairwise", {}).get("learned_vs_zeroshot", {})
        opd_pair = summary.get("pairwise", {}).get(
            "learned_opd_vs_baseline", {})
        rows.append({
            "run_dir": str(summary_path.parent),
            "scale_arguments": manifest.get("arguments", {}),
            "split_seed": manifest.get("evaluation", {}).get("split_seed"),
            "visual_backend": manifest.get("encoders", {}).get("visual"),
            "verdict": summary.get("verdict"),
            "learned_f1": learned.get("f1_valid"),
            "learned_judge_accuracy": learned.get("judge_accuracy"),
            "learned_avg_cost": learned.get("avg_cost"),
            "learned_gold_page_selection_rate": learned.get(
                "gold_page_any_selection_rate"),
            "learned_vs_zeroshot": pair,
            "opd_enabled": manifest.get("training", {}).get("opd_enabled"),
            "learned_opd_vs_baseline": opd_pair,
        })

    aggregates = aggregate_by_scale(rows)
    (campaign_root / "campaign_summary.json").write_text(
        json.dumps({"runs": rows, "aggregates": aggregates},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# File_Router experiment campaign", "",
        f"Runs: {len(rows)}", "",
        "| Run | Seed | Visual | OPD | Verdict | Learned F1 | Avg Cost | vs Zeroshot Saving | OPD vs Base Saving | OPD vs Base F1 |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pair = row.get("learned_vs_zeroshot") or {}
        opd_pair = row.get("learned_opd_vs_baseline") or {}
        lines.append(
            f"| `{Path(row['run_dir']).name}` | {row.get('split_seed')} | "
            f"{row.get('visual_backend')} | {row.get('opd_enabled')} | "
            f"{row.get('verdict')} | {row.get('learned_f1')} | "
            f"{row.get('learned_avg_cost')} | {pair.get('cost_saving_rate')} | "
            f"{opd_pair.get('cost_saving_rate')} | "
            f"{opd_pair.get('f1_delta_valid')} |")
    if aggregates:
        lines += ["", "## Across-seed summary", "",
                  "Read multi-seed results as mean +/- sd; a single seed cannot show whether "
                  "an improvement is robust.", ""]
        for entry in aggregates:
            seeds = ", ".join(str(seed) for seed in entry["seeds"])
            lines += [
                f"### subset={entry['subset']} / {entry['visual_backend']} "
                f"({entry['runs']} runs; seeds {seeds})", "",
                "| Metric | Mean | Sd | Min | Max |",
                "|---|---:|---:|---:|---:|",
            ]
            for key, label in AGGREGATED_METRICS:
                stat = entry["metrics"].get(key)
                if not stat:
                    continue
                lines.append(
                    f"| {label} | {stat['mean']} | {stat['std']} | "
                    f"{stat['min']} | {stat['max']} |")
            lines.append("")
    (campaign_root / "CAMPAIGN_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    root = Path(args.campaign_root)
    if not root.exists():
        parser.error(f"campaign root does not exist: {root}")
    rows = summarize(root)
    print(f"[SUMMARY] runs={len(rows)}")
    print(f"[SUMMARY] report={root / 'CAMPAIGN_REPORT.md'}")


if __name__ == "__main__":
    main()
