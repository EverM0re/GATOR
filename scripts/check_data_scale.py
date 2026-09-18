"""Fail fast when a requested experiment would reuse an undersized dataset.

The validation campaign supports ``--skip-download`` so multiple seeds can
reuse the same raw data and indexes.  Reusing a smoke subset for a medium or
large run, however, silently produces a misleading experiment.  This check is
run after unified data is built and before the expensive ingest stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config  # noqa: E402
from file_router.data.evidence import (  # noqa: E402
    has_visual_modality,
    page_modalities,
)


# Known public split limits.  They prevent a valid large run from requiring
# more rows than a source actually contains (UniDoc finance has 200 QA and the
# ViDoRe test parquet has 500 rows).
# Ceilings imposed by the public sources themselves, so a large run never
# demands more rows than exist.  UniDoc's finance split yields 157 QA after
# document-level filtering; requiring 200 made every large run fail a check it
# could not have satisfied.
SOURCE_QA_CAPS = {
    "unidoc": 157,
    "vidore": 500,
}
IMAGE_REQUIRED = {"unidoc", "mmdocrag", "vidore", "real_mm_rag"}


def minimum_qa(dataset: str, subset: int, docs: int,
               eval_per_dataset: int) -> int:
    """Return a conservative lower bound for a meaningful requested scale."""
    source_target = min(subset, SOURCE_QA_CAPS.get(dataset, subset))
    if subset <= 50:
        # Smoke deliberately limits document-backed datasets to very few docs.
        return max(1, min(source_target, docs or source_target))
    # Medium/large need enough rows for both training and evaluation.  This is
    # intentionally below the requested cap because document limits and public
    # source sizes can make the exact cap unreachable.
    return max(1, min(source_target, max(docs, eval_per_dataset * 2)))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def inspect_unified(unified_dir: str | Path, dataset: str) -> dict:
    root = Path(unified_dir) / dataset
    qa = _read_jsonl(root / "qa.jsonl")
    corpus = _read_jsonl(root / "corpus.jsonl")
    image_docs = 0
    image_pages = set()
    for document in corpus:
        pages = document.get("pages") or []
        if any(str(page.get("image_path") or "").strip() for page in pages):
            image_docs += 1
        for page in pages:
            if str(page.get("image_path") or "").strip():
                image_pages.add((document.get("doc_id"), page.get("page_num")))
    evidence_qa = 0
    page_modality_qa = 0
    visual_gold_qa = 0
    visual_gold_with_images = 0
    for question in qa:
        evidence = question.get("evidence") or {}
        evidence_pages = evidence.get("pages") or []
        if not evidence_pages:
            continue
        evidence_qa += 1
        if evidence.get("page_modalities"):
            page_modality_qa += 1
        visual_pages = [
            page for page in evidence_pages
            if has_visual_modality(page_modalities(evidence, page))
        ]
        if visual_pages:
            visual_gold_qa += 1
            if all((question.get("doc_id"), page) in image_pages
                   for page in visual_pages):
                visual_gold_with_images += 1
    return {
        "dataset": dataset,
        "qa": len(qa),
        "docs": len(corpus),
        "image_docs": image_docs,
        "evidence_qa": evidence_qa,
        "page_modality_qa": page_modality_qa,
        "visual_gold_qa": visual_gold_qa,
        "visual_gold_with_images": visual_gold_with_images,
    }


def validate_scale(cfg, subset: int, docs: int,
                   eval_per_dataset: int) -> tuple[list[dict], list[str]]:
    datasets = list(cfg.datasets.enabled) + list(cfg.datasets.retrieval_only)
    rows = []
    problems = []
    for dataset in datasets:
        row = inspect_unified(cfg.paths.unified_dir, dataset)
        required = minimum_qa(dataset, subset, docs, eval_per_dataset)
        row["minimum_qa"] = required
        rows.append(row)
        if row["qa"] < required:
            problems.append(
                f"{dataset}: only {row['qa']} QA, need at least {required}")
        if dataset in IMAGE_REQUIRED:
            required_image_docs = min(row["docs"], required)
            required_image_docs = max(1, math.ceil(required_image_docs * 0.8))
            if row["image_docs"] < required_image_docs:
                problems.append(
                    f"{dataset}: only {row['image_docs']}/{row['docs']} documents "
                    f"have usable images, need at least {required_image_docs}")
        if dataset == "mmdocrag" and row["qa"] >= required:
            if row["evidence_qa"] and row["page_modality_qa"] < row["evidence_qa"]:
                problems.append(
                    "mmdocrag: Gold quote-level page_modalities missing for "
                    f"{row['evidence_qa'] - row['page_modality_qa']}/"
                    f"{row['evidence_qa']} evidence QA")
            required_visual_images = math.ceil(row["visual_gold_qa"] * 0.9)
            if (row["visual_gold_qa"] == 0 or
                    row["visual_gold_with_images"] < required_visual_images):
                problems.append(
                    "mmdocrag: visual Gold image coverage is "
                    f"{row['visual_gold_with_images']}/{row['visual_gold_qa']}, "
                    f"need at least {required_visual_images}")
    return rows, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/file_router.yaml")
    parser.add_argument("--subset", type=int, required=True)
    parser.add_argument("--docs", type=int, required=True)
    parser.add_argument("--eval-per-dataset", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows, problems = validate_scale(
        cfg, args.subset, args.docs, args.eval_per_dataset)

    print("[DATA SCALE] unified data preflight", flush=True)
    print("dataset          QA/min     docs   image_docs  visual_gold/images", flush=True)
    for row in rows:
        print(
            f"{row['dataset']:<16} {row['qa']:>4}/{row['minimum_qa']:<4} "
            f"{row['docs']:>7} {row['image_docs']:>12} "
            f"{row['visual_gold_qa']:>7}/{row['visual_gold_with_images']:<7}",
            flush=True,
        )
    if problems:
        print("[FAIL] The available data is too small or incomplete:", flush=True)
        for problem in problems:
            print(f"  - {problem}", flush=True)
        print(
            "Re-run this scale with downloads enabled "
            "(SKIP_DOWNLOAD=0 or MEDIUM_SKIP_DOWNLOAD=0).",
            flush=True,
        )
        return 2
    print("[PASS] unified data matches the requested experiment scale", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
