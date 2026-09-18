"""Flatten unified corpus/QA into the per-record format host adapters ingest.

Hosts take flat text records, not the nested {doc_id, pages:[...]} shape the
pipeline uses internally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# node_class -> cost tier, matching TIER_NAMES in evaluate_validation.
_TIER_OF_CLASS = {
    "doc_summary": "caption",
    "text_span": "fulltext",
    "pdf_page": "screenshot",
}


def flatten_from_store(store_dir: Path, dataset: str, out_dir: Path,
                       max_docs: int = 0) -> dict:
    """Read ingested nodes out of store/<dataset>/nodes.sqlite.

    Preferred over the unified corpus whenever ingestion added text that the
    unified files never had: UniDoc pages carry no text layer, so their text
    only exists after OCR runs during ingest, and it is written to the node
    store rather than back to data/unified.

    Emits one record per node with its ingestion-time cost, so host adapters can
    price candidates with the same numbers the pipeline uses instead of
    re-estimating from character counts.
    """
    import sqlite3

    db = store_dir / dataset / "nodes.sqlite"
    if not db.exists():
        raise FileNotFoundError(str(db))

    out_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db))
    records, seen_docs = [], set()
    for (blob,) in connection.execute("SELECT blob FROM nodes"):
        node = json.loads(blob)
        text = (node.get("text") or "").strip()
        if not text:
            continue
        ref = node.get("source_ref") or {}
        doc_id = ref.get("doc_id") or ""
        if max_docs:
            seen_docs.add(doc_id)
            if len(seen_docs) > max_docs:
                break
        records.append({
            "doc_id": doc_id,
            "page": ref.get("page_num"),
            "text": text,
            "image_path": node.get("image_path") or node.get("page_image_path") or "",
            "tier": _TIER_OF_CLASS.get(node.get("node_class"), node.get("node_class")),
            "cost": node.get("token_equivalent_cost"),
            "node_id": node.get("node_id"),
            # Tiers of one page share this, so the router treats them as the
            # same evidence at different prices.
            "group_key": node.get("redundancy_cluster_id") or f"{doc_id}:p{ref.get('page_num')}",
        })
    connection.close()

    corpus_path = out_dir / f"{dataset}_corpus_flat.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    tiers: dict = {}
    for record in records:
        tiers[record["tier"]] = tiers.get(record["tier"], 0) + 1
    return {"dataset": dataset, "source": "store", "records": len(records),
            "tiers": tiers, "corpus": str(corpus_path)}


def _write_questions(unified_dir: Path, dataset: str, out_dir: Path,
                     info: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = []
    with open(unified_dir / dataset / "qa.jsonl", encoding="utf-8") as handle:
        for line in handle:
            qa = json.loads(line)
            questions.append({
                "qa_id": qa.get("qa_id", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "doc_id": qa.get("qrel_doc_id") or qa.get("doc_id", ""),
            })
    path = out_dir / f"{dataset}_questions.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for qa in questions:
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")
    info["questions"] = len(questions)
    info["questions_path"] = str(path)


def flatten(unified_dir: Path, dataset: str, out_dir: Path,
            max_docs: int = 0, require_text: bool = True) -> dict:
    src = unified_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = [], 0
    with open(src / "corpus.jsonl", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_docs and index >= max_docs:
                break
            doc = json.loads(line)
            for page in doc.get("pages", []):
                text = (page.get("text") or "").strip()
                caption = (page.get("caption") or "").strip()
                body = text or caption
                if require_text and not body:
                    # Pages whose text lives only in the page image (UniDoc
                    # before OCR) cannot be ingested by text-only hosts.
                    skipped += 1
                    continue
                records.append({
                    "doc_id": doc.get("doc_id", ""),
                    "page": page.get("page_num"),
                    "text": body,
                    "image_path": page.get("image_path") or "",
                })

    corpus_path = out_dir / f"{dataset}_corpus_flat.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    questions = []
    with open(src / "qa.jsonl", encoding="utf-8") as handle:
        for line in handle:
            qa = json.loads(line)
            questions.append({
                "qa_id": qa.get("qa_id", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "doc_id": qa.get("qrel_doc_id") or qa.get("doc_id", ""),
            })
    questions_path = out_dir / f"{dataset}_questions.jsonl"
    with open(questions_path, "w", encoding="utf-8") as handle:
        for qa in questions:
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")

    return {"dataset": dataset, "records": len(records),
            "skipped_no_text": skipped, "questions": len(questions),
            "corpus": str(corpus_path), "questions_path": str(questions_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-dir", default="data/unified")
    parser.add_argument("--datasets", nargs="+",
                        default=["mmdocrag", "unidoc"],
                        help="unified-corpus datasets to flatten; memgallery "
                             "requires download_memgallery.py first")
    parser.add_argument("--out-dir", default="data/mount")
    parser.add_argument("--store-dir", default="store")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--source", choices=["auto", "store", "unified"],
                        default="auto",
                        help="auto: use the node store when it has text, else "
                             "fall back to the unified corpus")
    args = parser.parse_args()

    for dataset in args.datasets:
        info = None
        if args.source in ("auto", "store"):
            try:
                info = flatten_from_store(Path(args.store_dir), dataset,
                                          Path(args.out_dir), args.max_docs)
                if info["records"] == 0 and args.source == "auto":
                    print(f"[prep] {dataset}: store has no text; "
                          "falling back to unified")
                    info = None
            except FileNotFoundError as exc:
                if args.source == "store":
                    print(f"[prep] {dataset}: missing {exc}")
                    continue
                print(f"[prep] {dataset}: no node store; using unified corpus")

        if info is None:
            try:
                info = flatten(Path(args.unified_dir), dataset,
                               Path(args.out_dir), args.max_docs)
            except FileNotFoundError as exc:
                print(f"[prep] {dataset}: missing {exc.filename}")
                continue

        # Questions always come from the unified split; the store holds evidence,
        # not QA.
        try:
            _write_questions(Path(args.unified_dir), dataset, Path(args.out_dir),
                             info)
        except FileNotFoundError as exc:
            print(f"[prep] {dataset}: missing {exc.filename}")

        print(json.dumps(info, ensure_ascii=False))
        if info["records"] == 0:
            print(f"[prep] WARNING: {dataset} produced no records. "
                  "UniDoc pages have no text layer, so their text only exists "
                  "after an ingest with ingestion.ocr.enabled=true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
