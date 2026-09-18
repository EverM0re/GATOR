"""Fetch Mem-Gallery and convert it into this project's unified format.

Mem-Gallery (Bei et al., ACL 2026; github.com/YuanchenBei/Mem-Gallery) is a
multimodal long-term conversational memory benchmark. It matters here because it
ships page images as local files rather than hot-linked URLs, so it exercises
the screenshot rung on data this project did not build.

There is no pip package, so this pulls the dataset from the Hugging Face mirror
rather than cloning the harness -- we need the corpus, not their runner.

    python3 -m scripts.download_memgallery --out data/unified/memgallery
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = "YuanchenBei/Mem-Gallery"


def _load(src: Path):
    """Read the benchmark file, whichever of the shipped layouts is present."""
    for name in ("data.json", "mem_gallery.json", "benchmark.json"):
        path = src / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    hits = sorted(src.glob("*.json")) + sorted(src.glob("**/*.json"))
    if not hits:
        raise FileNotFoundError(
            f"no .json found under {src}. Download the dataset first:\n"
            f"  huggingface-cli download {REPO} --repo-type dataset "
            f"--local-dir {src}")
    return json.loads(hits[0].read_text(encoding="utf-8"))


def convert(src: Path, out: Path) -> dict:
    raw = _load(src)
    sessions = raw if isinstance(raw, list) else (
        raw.get("sessions") or raw.get("data") or [])
    out.mkdir(parents=True, exist_ok=True)

    docs, qas, missing_images = [], [], 0
    for s_i, session in enumerate(sessions):
        doc_id = str(session.get("session_id") or session.get("id") or s_i)
        pages = []
        turns = (session.get("turns") or session.get("dialogue")
                 or session.get("conversation") or [])
        for t_i, turn in enumerate(turns):
            image = (turn.get("image") or turn.get("img_path")
                     or turn.get("image_path") or "")
            if image:
                candidate = (src / image) if not Path(image).is_absolute() \
                    else Path(image)
                if candidate.exists():
                    image = str(candidate)
                else:
                    # Local files are the reason to use this corpus; a missing
                    # one silently degrades the screenshot rung to text.
                    missing_images += 1
                    image = ""
            text = (turn.get("text") or turn.get("content")
                    or turn.get("utterance") or "")
            speaker = turn.get("speaker") or turn.get("role") or ""
            pages.append({"page_num": t_i,
                          "text": f"{speaker}: {text}".strip(": "),
                          "image_path": image})
        docs.append({"doc_id": doc_id, "pages": pages})

        for qa in (session.get("qa") or session.get("questions") or []):
            qas.append({
                "qa_id": str(qa.get("qa_id") or qa.get("id")
                             or f"{doc_id}_{len(qas)}"),
                "question": qa.get("question") or qa.get("query") or "",
                "answer": str(qa.get("answer") or qa.get("gold_answer") or ""),
                "qrel_doc_id": doc_id,
                "evidence": {"pages": qa.get("evidence_turns")
                             or qa.get("clue_ids") or []},
            })

    with open(out / "corpus.jsonl", "w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with open(out / "qa.jsonl", "w", encoding="utf-8") as handle:
        for qa in qas:
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")

    with_images = sum(1 for d in docs for p in d["pages"] if p["image_path"])
    return {"documents": len(docs), "questions": len(qas),
            "pages_with_images": with_images,
            "images_missing_on_disk": missing_images}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/memgallery",
                    help="directory holding the downloaded Mem-Gallery files")
    ap.add_argument("--out", default="data/unified/memgallery")
    args = ap.parse_args()

    try:
        stats = convert(Path(args.src), Path(args.out))
    except FileNotFoundError as exc:
        print(f"[memgallery] {exc}", file=sys.stderr)
        return 2

    print(f"[memgallery] {stats}")
    if stats["documents"] == 0 or stats["questions"] == 0:
        print("[memgallery] parsed nothing usable -- the released layout "
              "differs from the fields this script expects; inspect the JSON "
              "and adjust convert().", file=sys.stderr)
        return 3
    if stats["pages_with_images"] == 0:
        print("[memgallery] WARNING: no local images resolved. This corpus was "
              "chosen for its local page images; without them it exercises "
              "only the text rungs.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
