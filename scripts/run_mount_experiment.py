"""Mount File_Router's selector on a host memory system and A/B it.

Controlled comparison, per query:
  A (control) the host's own retrieval, sent verbatim -- what it does today
  B (routed)  the SAME retrieval, re-priced by CostRouter

The host's retriever is never replaced, so any difference is attributable to
granularity selection alone.  When an answer model is configured both arms are
answered and scored, because a cost reduction that silently loses accuracy is
not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.evaluation.metrics import answer_metrics  # noqa: E402
from file_router.plugin import CostRouter  # noqa: E402
from integrations.adapters import ADAPTERS  # noqa: E402


def _load_jsonl(path: Path, limit: int = 0) -> List[dict]:
    if not path.exists():
        raise SystemExit(
            f"[mount] missing {path}\n"
            "[mount] this file is generated, and re-uploading File_Router wipes it.\n"
            "[mount] regenerate it with:\n"
            "[mount]   python3 -m scripts.prepare_mount_data "
            "--datasets mmdocrag unidoc")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _answer(vlm, question: str, context: str) -> str:
    if vlm is None:
        return ""
    try:
        return vlm.generate_answer(question, context=context)
    except Exception as exc:  # noqa: BLE001
        print(f"[mount] answer failed: {repr(exc)[:120]}")
        return ""


def run(args) -> Dict[str, Any]:
    adapter_cls = ADAPTERS[args.host]
    router = CostRouter.with_defaults(
        max_groups=args.max_groups,
        min_distinct_pages=args.min_distinct_pages,
    )
    # Hosts resolve their LLM endpoint from the same config the pipeline uses,
    # so there is no second place to configure it.
    try:
        adapter = adapter_cls(router=router, top_k=args.top_k,
                              config_path=args.config)
    except TypeError:
        # Adapters that take no endpoint config (e.g. a running MemVerse server).
        adapter = adapter_cls(router=router, top_k=args.top_k)

    corpus = _load_jsonl(Path(args.corpus), args.max_docs)
    print(f"[mount] host={args.host} ingesting {len(corpus)} records")
    ingested, first_error = 0, None
    started_ingest = time.perf_counter()
    for index, record in enumerate(corpus, 1):
        try:
            adapter.add(record)
            ingested += 1
        except Exception as exc:  # noqa: BLE001
            if first_error is None:
                first_error = repr(exc)
                print(f"[mount] ingest failed: {first_error[:200]}")
            break
        # A host that calls an LLM per record (A-Mem has no infer=False escape)
        # can silently consume hours.  Stop and evaluate on what was ingested
        # rather than burning a whole overnight slot on one phase.
        if args.ingest_timeout > 0:
            spent = time.perf_counter() - started_ingest
            if spent > args.ingest_timeout:
                print(f"[mount] ingest timeout after {spent / 60:.1f} min; "
                      f"continuing with {ingested}/{len(corpus)} records")
                break
        # Ingest dominates wall-clock on a large corpus, so report a rate and an
        # ETA rather than appearing to hang.
        if index % 50 == 0:
            elapsed = time.perf_counter() - started_ingest
            rate = index / max(elapsed, 1e-6)
            remaining = (len(corpus) - index) / max(rate, 1e-6)
            print(f"[mount] ingested {index}/{len(corpus)} "
                  f"({rate:.1f}/s, ~{remaining / 60:.1f} min left)")
    print(f"[mount] ingested {ingested}/{len(corpus)} in "
          f"{time.perf_counter() - started_ingest:.1f}s")
    # An empty store makes every retrieval return nothing, which would still
    # produce a "100% cost saving" table.  Refuse to report that as a result.
    if ingested == 0:
        raise SystemExit(
            "[mount] ABORT: nothing was ingested, so the host has no memory to "
            "retrieve from and any comparison would be meaningless.\n"
            f"[mount] first error: {first_error}\n"
            "[mount] common cause: the host is calling OpenAI instead of your "
            "local endpoint. Set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY.")
    if ingested < len(corpus):
        print(f"[mount] WARNING: only {ingested}/{len(corpus)} records ingested; "
              "results describe a partial corpus")

    vlm = None
    if args.answer:
        from file_router.config import load_config
        from file_router.encoders import VLM

        vlm = VLM(load_config(args.config).llm)

    questions = _load_jsonl(Path(args.questions), 0)
    if args.question_seed:
        # Sample a different subset per seed so repeated runs measure variation
        # across question samples, not the same questions twice.  Without this a
        # "multi-seed" mount result would be the identical run relabelled.
        random.Random(args.question_seed).shuffle(questions)
    if args.max_questions:
        questions = questions[:args.max_questions]
    print(f"[mount] evaluating {len(questions)} questions"
          + (f" (seed={args.question_seed})" if args.question_seed else ""))

    rows: List[Dict[str, Any]] = []
    empty_retrievals = 0
    for index, qa in enumerate(questions, 1):
        comparison = adapter.compare(qa["question"])
        if not comparison.baseline_ids:
            empty_retrievals += 1
        row = comparison.as_dict()
        row["qa_id"] = qa.get("qa_id", f"q{index}")

        if vlm is not None:
            gold = qa.get("answer", "")
            base_answer = _answer(vlm, qa["question"], comparison.baseline_text)
            routed_answer = _answer(vlm, qa["question"], comparison.routed_text)
            row["baseline_f1"] = answer_metrics(base_answer, gold)["token_f1"]
            row["routed_f1"] = answer_metrics(routed_answer, gold)["token_f1"]
            row["baseline_answer"] = base_answer
            row["routed_answer"] = routed_answer
        rows.append(row)
        if index % 10 == 0:
            print(f"[mount] {index}/{len(questions)}")

    if empty_retrievals == len(rows) and rows:
        raise SystemExit(
            "[mount] ABORT: the host returned no candidates for any question. "
            "Cost numbers from an empty retrieval are not a result.")
    if empty_retrievals:
        print(f"[mount] WARNING: {empty_retrievals}/{len(rows)} questions "
              "retrieved nothing")

    summary = summarize(args.host, rows)
    # Record the requested depth: a host may return more candidates than asked
    # (family expansion), and labelling a run by what came back rather than what
    # was requested makes an aligned sweep look like three different settings.
    summary["top_k"] = args.top_k
    summary["ingested"] = ingested
    summary["corpus_size"] = len(corpus)
    summary["empty_retrievals"] = empty_retrievals
    return summary


def summarize(host: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(key: str) -> float:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return statistics.mean(values) if values else 0.0

    baseline_cost, routed_cost = mean("baseline_cost"), mean("routed_cost")
    summary: Dict[str, Any] = {
        "host": host,
        "n": len(rows),
        "baseline_cost": round(baseline_cost, 2),
        "routed_cost": round(routed_cost, 2),
        "cost_saving_rate": round(
            (baseline_cost - routed_cost) / baseline_cost, 4)
        if baseline_cost else 0.0,
        "avg_candidates": round(mean("candidate_count"), 2),
        "avg_kept": round(mean("kept_count"), 2),
        "avg_retrieval_ms": round(mean("retrieval_ms"), 2),
        "avg_routing_ms": round(mean("routing_ms"), 2),
    }
    if any("baseline_f1" in r for r in rows):
        baseline_f1, routed_f1 = mean("baseline_f1"), mean("routed_f1")
        # Paired statistics: the two arms answer the SAME questions from the
        # SAME retrieval, so the per-question difference is the right unit.  A
        # mean delta without a CI cannot distinguish "quality preserved" from
        # "too few samples to tell", and that distinction is the whole claim.
        paired = [r["routed_f1"] - r["baseline_f1"] for r in rows
                  if "baseline_f1" in r and "routed_f1" in r]
        if len(paired) > 1:
            spread = statistics.stdev(paired)
            error = spread / (len(paired) ** 0.5)
            summary.update({
                "f1_delta_ci95": [round(statistics.mean(paired) - 1.96 * error, 4),
                                  round(statistics.mean(paired) + 1.96 * error, 4)],
                "f1_delta_std": round(spread, 4),
                "n_better": sum(1 for d in paired if d > 0),
                "n_worse": sum(1 for d in paired if d < 0),
                "n_tied": sum(1 for d in paired if d == 0),
            })
        summary.update({
            "baseline_f1": round(baseline_f1, 4),
            "routed_f1": round(routed_f1, 4),
            "f1_delta": round(routed_f1 - baseline_f1, 4),
            # The claim is "cheaper at equal quality", so efficiency is the
            # number that carries it; cost alone can always be won by sending
            # less and answering worse.
            "baseline_f1_per_1k": round(1000 * baseline_f1 / baseline_cost, 4)
            if baseline_cost else 0.0,
            "routed_f1_per_1k": round(1000 * routed_f1 / routed_cost, 4)
            if routed_cost else 0.0,
        })
    summary["rows"] = rows
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--corpus", required=True,
                        help="jsonl with {doc_id, text, image_path?}")
    parser.add_argument("--questions", required=True,
                        help="jsonl with {qa_id, question, answer}")
    parser.add_argument("--config", default="config/file_router.yaml")
    parser.add_argument("--out", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-groups", type=int, default=5)
    parser.add_argument("--min-distinct-pages", type=int, default=4)
    parser.add_argument("--question-seed", type=int, default=0,
                        help="shuffle questions with this seed before taking "
                             "--max-questions, so repeated runs sample "
                             "different subsets (0 = keep file order)")
    parser.add_argument("--ingest-timeout", type=float, default=0,
                        help="seconds; stop ingesting and evaluate on what was "
                             "loaded (0 = no limit). Useful for hosts that call "
                             "an LLM per record.")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="cap ingested corpus records; use a small value "
                             "for smoke tests (full mmdocrag is ~8k records)")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--answer", action="store_true",
                        help="also generate and score answers for both arms")
    args = parser.parse_args()

    started = time.perf_counter()
    summary = run(args)
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)

    # Default into the migrated results directory, not the repo root: anything
    # left in the repo is destroyed by a full-folder upload, which is exactly
    # how the first 50-question mem0 result was lost.
    out = Path(args.out) if args.out else (
        Path("experiment_campaigns") / "mounts" / f"mount_{args.host}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"[mount] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
