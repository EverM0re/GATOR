"""Stage 4: ingest each dataset's corpus.jsonl into its own store/<dataset>/.

Builds the 3-tier cost ladder (caption / screenshot / full-text) per page.

    python -m scripts.ingest --config config/file_router.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config         # noqa: E402
from file_router.memory import SharedEncoders, DatasetMemory  # noqa: E402
from file_router.utils import banner, info, warn    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/file_router.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    # This stage deletes and rebuilds each dataset's store, so a run that must
    # not disturb an existing one needs its own directory.  Without this, a
    # full-corpus ingest would destroy the store the reported results came from.
    if os.environ.get("STORE_DIR"):
        cfg.paths.store_dir = os.environ["STORE_DIR"]
        os.makedirs(cfg.paths.store_dir, exist_ok=True)
        print(f"[ingest] store_dir overridden -> {cfg.paths.store_dir}")

    targets = args.only or (list(cfg.datasets.enabled) + list(cfg.datasets.retrieval_only))
    banner(f"INGEST datasets={targets}")
    enc = SharedEncoders(cfg)
    for ds in targets:
        corpus = os.path.join(cfg.paths.unified_dir, ds, "corpus.jsonl")
        if not os.path.exists(corpus):
            warn(f"[ingest] no corpus for {ds}; skip"); continue
        info(f"--- ingesting {ds} ---")
        # fresh store each run so re-ingest never double-counts the indexes
        store_dir = os.path.join(cfg.paths.store_dir, ds)
        if os.path.exists(store_dir):
            shutil.rmtree(store_dir)
        mem = DatasetMemory(cfg, ds, enc)
        mem.ingestor().ingest_corpus(corpus)
    usage_path = os.path.join(cfg.paths.log_dir, "ingest_vlm_usage.json")
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(enc.vlm.usage_summary(), f, ensure_ascii=False, indent=2)
    info(f"[ingest] VLM token/time usage -> {usage_path}")
    info("ingest stage done.")


if __name__ == "__main__":
    main()
