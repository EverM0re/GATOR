"""Stage 3: unify raw downloads -> per-dataset corpus.jsonl/qa.jsonl,
then write reproducible train/test splits and merged train.jsonl/test.jsonl.

    python -m scripts.build_unified --config config/file_router.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config            # noqa: E402
from file_router.data.loaders import build_all         # noqa: E402
from file_router.data.split import choose_test_groups  # noqa: E402
from file_router.utils import banner, info, warn       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/file_router.yaml")
    # download_data takes --docs, but the unified build read the cap only from
    # the config, so `--docs 80` fetched 80 documents and this stage then threw
    # all but subset_docs (12) of them away. The two stages must agree.
    ap.add_argument("--docs", type=int, default=None,
                    help="max documents per dataset; overrides "
                         "datasets.subset_docs in the config")
    ap.add_argument("--subset", type=int, default=None,
                    help="max QA per dataset; overrides datasets.subset_qa")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.docs is not None:
        cfg.datasets.subset_docs = args.docs
    if args.subset is not None:
        cfg.datasets.subset_qa = args.subset

    banner("BUILD UNIFIED FORMAT")
    build_all(cfg)

    banner("SPLIT train/test")
    os.makedirs(cfg.paths.splits_dir, exist_ok=True)
    all_train, all_test = [], []
    targets = list(cfg.datasets.enabled) + list(cfg.datasets.retrieval_only)
    for ds in targets:
        qa_path = os.path.join(cfg.paths.unified_dir, ds, "qa.jsonl")
        if not os.path.exists(qa_path):
            warn(f"[split] missing {qa_path}; skip"); continue
        rows = []
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        test_groups = choose_test_groups(
            (rec.get("doc_id") or rec["qa_id"] for rec in rows),
            cfg.split.test_ratio,
            cfg.split.seed,
        )
        train_ids, test_ids = [], []
        for rec in rows:
            group_id = rec.get("doc_id") or rec["qa_id"]
            if group_id in test_groups:
                test_ids.append(rec["qa_id"]); all_test.append(rec)
            else:
                train_ids.append(rec["qa_id"]); all_train.append(rec)
        with open(os.path.join(cfg.paths.splits_dir, f"{ds}.train.txt"), "w") as f:
            f.write("\n".join(train_ids) + ("\n" if train_ids else ""))
        with open(os.path.join(cfg.paths.splits_dir, f"{ds}.test.txt"), "w") as f:
            f.write("\n".join(test_ids) + ("\n" if test_ids else ""))
        info(f"[split] {ds}: train={len(train_ids)} test={len(test_ids)}")

    with open(os.path.join(cfg.paths.unified_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in all_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(cfg.paths.unified_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for r in all_test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info(f"[split] merged train={len(all_train)} test={len(all_test)} "
         f"-> {cfg.paths.unified_dir}/train.jsonl,test.jsonl")


if __name__ == "__main__":
    main()
