"""Render experiment outputs as one categorised markdown report.

Tables over prose by design: each section is a category of metric, so numbers
can be compared down a column without reading paragraphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve(pattern: str) -> List[Path]:
    """Expand a path that may contain a glob.

    Run directories are timestamped, so callers naturally write
    `runs/validation_*/summary.json`.  When the shell does not expand that (a
    quoted argument, or a symlinked parent), the literal reaches us and every
    input silently goes missing -- so expand it here instead of trusting the
    shell.  Newest match wins.
    """
    if any(ch in pattern for ch in "*?["):
        matches = sorted(Path().glob(pattern))
        return matches[-1:] if matches else []
    return [Path(pattern)]


def _read(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _mount_label(payload: dict, path: Path) -> str:
    """A readable identity for a mount run.

    A dozen rows all reading "mem0" tell a reader nothing, so derive the corpus
    and retrieval depth from the run itself, falling back to the filename.
    """
    stem = path.stem.lower()
    corpus = next((n for n in ("mmdocrag", "unidoc", "locomo") if n in stem), None)
    depth = payload.get("top_k") or payload.get("avg_candidates")
    parts = [corpus] if corpus else []
    if depth:
        parts.append(f"k={int(round(float(depth)))}")
    return " ".join(parts) if parts else path.stem


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _delta(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.4f}"


# ------------------------------------------------------------------ standalone
def standalone_section(summary: dict) -> List[str]:
    overall = summary.get("routers", {}).get("overall", {})
    order = [r for r in ("oracle", "full", "rule", "zeroshot",
                         "learned_baseline", "learned") if r in overall]

    lines = ["## 1. Standalone results", "",
             "### 1.1 Quality", "",
             "| Router | N | F1 | EM | Judge | UNKNOWN |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in order:
        r = overall[name]
        lines.append(
            f"| {name} | {r.get('n', '—')} | {_num(r.get('f1_valid'))} | "
            f"{_num(r.get('em_valid'))} | {_num(r.get('judge_accuracy'))} | "
            f"{_pct(r.get('unknown_answer_rate'))} |")

    lines += ["", "### 1.2 Cost and efficiency", "",
              "| Router | Avg Cost | P90 | Groups kept | **F1/1k tokens** |",
              "|---|---:|---:|---:|---:|"]
    for name in order:
        r = overall[name]
        lines.append(
            f"| {name} | {_num(r.get('avg_cost'), 1)} | {_num(r.get('p90_cost'), 1)} | "
            f"{_num(r.get('avg_selected_groups'), 2)} | "
            f"**{_num(r.get('f1_per_1k_tokens'))}** |")

    lines += ["", "### 1.3 Evidence selection", "",
              "| Router | Gold page hit | Gold doc hit | Tier distribution |",
              "|---|---:|---:|---|"]
    for name in order:
        r = overall[name]
        lines.append(
            f"| {name} | {_num(r.get('gold_page_any_selection_rate'))} | "
            f"{_num(r.get('gold_doc_selection_rate'))} | "
            f"`{json.dumps(r.get('tier_distribution') or {}, ensure_ascii=False)}` |")

    learned = overall.get("learned", {})
    lines += ["", "### 1.4 Candidate-pool recall (the router ceiling)", "",
              "| Metric | Value |", "|---|---:|",
              f"| Gold doc Recall@5 | {_num(learned.get('candidate_gold_doc_recall_at_5'))} |",
              f"| Gold page Recall (full pool) | {_num(learned.get('candidate_gold_page_recall'))} |",
              f"| Gold page Recall@5 | {_num(learned.get('candidate_gold_page_recall_at_5'))} |"]

    pairwise = summary.get("pairwise", {})
    if pairwise:
        lines += ["", "### 1.5 Paired comparisons", "",
                  "| Comparison | Cost saving | dF1 | F1 95% CI | dJudge | Strict Pareto |",
                  "|---|---:|---:|---|---:|---:|"]
        for key, p in pairwise.items():
            ci = p.get("f1_delta_ci95")
            lines.append(
                f"| {key} | {_pct(p.get('cost_saving_rate'))} | "
                f"{_delta(p.get('f1_delta_valid'))} | "
                f"`{ci}` | {_delta(p.get('judge_accuracy_delta'))} | "
                f"{_pct(p.get('strict_pareto_improvement_rate_valid'))} |")
    return lines


# ---------------------------------------------------------------------- mount
def mount_section(mounts: List[dict]) -> List[str]:
    lines = ["## 2. Mounting results (host vs. host + GATOR)", "",
             "The reference arm sends the host's own retrieval verbatim; the treatment arm re-prices",
             "**the same retrieval**. The retriever is unchanged, so any difference comes from granularity selection.", "",
             "### 2.1 Cost", "",
             "| Host | Config | N | Corpus | Host cost | Routed | Saving | Candidates | Kept |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in mounts:
        # A truncated ingest changes what the numbers mean, so it belongs in the
        # table rather than only in a log file.
        ingested, corpus = m.get("ingested"), m.get("corpus_size")
        coverage = "—"
        if ingested is not None and corpus:
            coverage = (f"{ingested}/{corpus}" if ingested >= corpus
                        else f"⚠️{ingested}/{corpus}")
        lines.append(
            f"| {m.get('host')} | {m.get('label', '')} | {m.get('n')} | "
            f"{coverage} | {_num(m.get('baseline_cost'), 1)} | "
            f"{_num(m.get('routed_cost'), 1)} | **{_pct(m.get('cost_saving_rate'))}** | "
            f"{_num(m.get('avg_candidates'), 1)} | {_num(m.get('avg_kept'), 1)} |")

    if any("baseline_f1" in m for m in mounts):
        lines += ["", "### 2.2 Quality (is the cost reduction paid for in accuracy?)", "",
                  "Both arms answer the same questions from the same retrieval, so the paired difference",
                  "is the correct statistical unit. An interval containing zero means no detectable change.", "",
                  "| Host | Host F1 | Routed F1 | ΔF1 | 95% CI | better/worse/tied | "
                  "Host F1/1k | Routed F1/1k |",
                  "|---|---:|---:|---:|---|---:|---:|---:|"]
        for m in mounts:
            ci = m.get("f1_delta_ci95")
            ci_text = f"`{ci}`" if ci else "—"
            counts = "—"
            if m.get("n_better") is not None:
                counts = (f"{m.get('n_better')}/{m.get('n_worse')}/"
                          f"{m.get('n_tied')}")
            lines.append(
                f"| {m.get('host')} | {_num(m.get('baseline_f1'))} | "
                f"{_num(m.get('routed_f1'))} | {_delta(m.get('f1_delta'))} | "
                f"{ci_text} | {counts} | "
                f"{_num(m.get('baseline_f1_per_1k'))} | "
                f"**{_num(m.get('routed_f1_per_1k'))}** |")

    weak = [m for m in mounts
            if isinstance(m.get("baseline_f1"), (int, float))
            and m["baseline_f1"] < 0.05]
    if weak:
        lines += ["",
                  "> ⚠️ The hosts below are too weak in absolute terms (baseline F1 < 0.05); both arms",
                  "> answer most questions identically. Such rows show only that routing did not",
                  "> make an already-ineffective retrieval worse; they are **not** evidence of quality preservation:",
                  ""]
        for m in weak:
            lines.append(f"> - `{m.get('host')}`: baseline F1 = "
                         f"{_num(m.get('baseline_f1'))}, "
                         f"identical answers {m.get('n_tied')}/{m.get('n')}")
        lines.append("")

    lines += ["", "### 2.3 Overhead", "",
              "| Host | Retrieval ms | Routing ms |", "|---|---:|---:|"]
    for m in mounts:
        lines.append(f"| {m.get('host')} | {_num(m.get('avg_retrieval_ms'), 1)} | "
                     f"{_num(m.get('avg_routing_ms'), 1)} |")
    return lines


# ------------------------------------------------------------- generalization
def generalization_section(matrix: Dict[str, Dict[str, dict]]) -> List[str]:
    tests = sorted({t for row in matrix.values() for t in row})
    lines = ["## 3. Cross-dataset generalization (train x test)", "",
             "Diagonal is in-domain, off-diagonal is transfer. Parentheses give the",
             "**training-free zeroshot** score on the same split.", ""]

    for label, key, zs_key in (("F1", "learned_f1", "zeroshot_f1"),
                               ("F1/1k tokens", "learned_f1_per_1k",
                                "zeroshot_f1_per_1k")):
        lines += [f"### 3.{1 if key.endswith('f1') else 2} {label}", "",
                  "| Train \\ Test | " + " | ".join(tests) + " |",
                  "|---|" + "---:|" * len(tests)]
        for train in sorted(matrix):
            cells = []
            for test in tests:
                cell = matrix[train].get(test)
                if not cell:
                    cells.append("—")
                    continue
                marker = "**" if train == test else ""
                zs = cell.get(zs_key)
                suffix = f" ({_num(zs)})" if zs is not None else ""
                cells.append(f"{marker}{_num(cell.get(key))}{marker}{suffix}")
            lines.append(f"| {train} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["### 3.3 Sample size (cells with n < 20 are not interpretable)", "",
              "| Train \\ Test | " + " | ".join(tests) + " |",
              "|---|" + "---:|" * len(tests)]
    for train in sorted(matrix):
        cells = []
        for test in tests:
            cell = matrix[train].get(test)
            if not cell:
                cells.append("—")
                continue
            n = cell.get("n")
            cells.append(f"{n}" if (n or 0) >= 20 else f"⚠️{n}")
        lines.append(f"| {train} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["### 3.4 Gold page hit rate", "",
              "| Train \\ Test | " + " | ".join(tests) + " |",
              "|---|" + "---:|" * len(tests)]
    for train in sorted(matrix):
        cells = []
        for test in tests:
            cell = matrix[train].get(test)
            cells.append(_num(cell.get("learned_gold_page")) if cell else "—")
        lines.append(f"| {train} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", default="",
                        help="a validation run's summary.json")
    parser.add_argument("--mount", nargs="*", default=[],
                        help="mount_<host>.json files")
    parser.add_argument("--generalization", default="",
                        help="generalization.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed-aggregate", default="",
                        help="campaign_summary.json holding cross-seed "
                             "mean/std for the standalone result")
    parser.add_argument("--auto", action="store_true",
                        help="find the newest results automatically")
    args = parser.parse_args()

    if args.auto:
        # Paths are timestamped and easy to mistype; discovering them removes a
        # class of "report written with an empty section" mistakes.
        if not args.standalone:
            runs = sorted(Path().glob(
                "experiment_campaigns/campaign_*/runs/validation_*/summary.json"))
            if runs:
                args.standalone = str(runs[-1])
                # A multi-seed campaign writes one summary per seed, so taking
                # the newest silently reports a single seed as if it were the
                # result.  Surface the campaign's own aggregate when one exists.
                campaign = Path(args.standalone).parents[2]
                siblings = list(campaign.glob("runs/validation_*/summary.json"))
                aggregate = campaign / "campaign_summary.json"
                if len(siblings) > 1 and aggregate.exists():
                    args.seed_aggregate = str(aggregate)
                    print(f"[report] {len(siblings)} seeds in {campaign.name}; "
                          f"aggregate: {aggregate}")
        if not args.seed_aggregate:
            aggregates = sorted(Path().glob(
                "experiment_campaigns/campaign_*/campaign_summary.json"))
            if aggregates:
                args.seed_aggregate = str(aggregates[-1])
        if not args.mount:
            args.mount = [str(p) for p in
                          sorted(Path().glob("experiment_campaigns/mounts/*.json"))]
        if not args.generalization:
            matrices = sorted(Path().glob(
                "experiment_campaigns/generalization*/generalization.json"))
            if matrices:
                args.generalization = str(matrices[-1])

    # Default the report into the migrated results directory so a full-folder
    # upload cannot delete it.
    if not args.out:
        args.out = ("experiment_campaigns/EXPERIMENT_REPORT.md"
                    if Path("experiment_campaigns").exists()
                    else "EXPERIMENT_REPORT.md")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    lines = ["# GATOR experiment report", ""]
    produced = []

    def seed_section(payload):
        """Cross-seed mean/std, which is what a multi-seed claim rests on."""
        out = ["", "### 1.6 Across seeds (standalone)", "",
               "A single seed cannot show robustness; read multi-seed results as mean +/- sd.", ""]
        for entry in payload.get("aggregates", []):
            seeds = ", ".join(str(x) for x in entry.get("seeds", []))
            out += [f"**subset={entry.get('subset')} / "
                    f"{entry.get('visual_backend')} "
                    f"({entry.get('runs')} runs; seeds {seeds})**", "",
                    "| Metric | Mean | Sd | Min | Max |",
                    "|---|---:|---:|---:|---:|"]
            for key, label in (("learned_f1", "Learned F1"),
                               ("learned_judge_accuracy", "Judge Acc"),
                               ("learned_gold_page_selection_rate", "Gold page hit"),
                               ("learned_avg_cost", "Avg cost"),
                               ("cost_saving_rate", "vs Zeroshot saving"),
                               ("f1_delta_valid", "vs Zeroshot F1 delta")):
                stat = entry.get("metrics", {}).get(key)
                if not stat:
                    continue
                out.append(f"| {label} | {_num(stat['mean'])} | "
                           f"{_num(stat['std'])} | {_num(stat['min'])} | "
                           f"{_num(stat['max'])} |")
            out.append("")
        return out

    standalone = None
    if args.standalone:
        found = _resolve(args.standalone)
        if found:
            print(f"[report] standalone: {found[0]}")
            standalone = _read(found[0])
        else:
            print(f"[report] MISSING: no file matches {args.standalone}")
    if standalone:
        lines += standalone_section(standalone) + [""]
        produced.append("standalone")
        if args.seed_aggregate:
            payload = _read(Path(args.seed_aggregate))
            if payload and payload.get("aggregates"):
                lines += seed_section(payload) + [""]
                produced.append("seed-aggregate")
            elif payload is None:
                print(f"[report] MISSING: {args.seed_aggregate}")

    # A missing input used to be filtered out silently, so a report could be
    # written with an entire section quietly absent -- and it still looked like
    # a successful run.  Name what could not be read.
    mount_paths: List[Path] = []
    for pattern in args.mount:
        found = _resolve(pattern)
        if not found:
            print(f"[report] MISSING: {pattern} (has that experiment been run?)")
            continue
        mount_paths.extend(found)

    mounts = []
    for path in mount_paths:
        payload = _read(path)
        if payload is None:
            print(f"[report] UNREADABLE: {path} (not valid JSON)")
            continue
        print(f"[report] mount: {path}")
        payload.setdefault("label", _mount_label(payload, path))
        mounts.append(payload)
    # Group by host, then corpus, then retrieval depth, so related rows sit
    # together and a trend across depth reads down the column.
    mounts.sort(key=lambda m: (m.get("host", ""), m.get("label", ""),
                               m.get("avg_candidates") or 0))
    if args.mount and not mounts:
        print("[report] none of the --mount files could be read; the mount "
              "section will be absent from the report")
    if mounts:
        lines += mount_section(mounts) + [""]
        produced.append("mount")

    matrix = None
    if args.generalization:
        found = _resolve(args.generalization)
        if found:
            matrix = _read(found[0])
        else:
            print(f"[report] MISSING: no file matches {args.generalization}")
    if matrix:
        lines += generalization_section(matrix) + [""]
        produced.append("generalization")

    if not produced:
        print("[report] no inputs found; nothing written")
        return 1

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    requested = [name for name, flag in (("standalone", args.standalone),
                                         ("mount", args.mount),
                                         ("generalization", args.generalization))
                 if flag]
    absent = [name for name in requested if name not in produced]
    print(f"[report] sections={produced} -> {args.out}")
    if absent:
        print(f"[report] WARNING: requested but missing: {absent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
