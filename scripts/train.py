"""Stage 5: train the router MLP on the TRAIN split.

Two phases:
  (1) collect: for each train QA, recall candidate groups, auto-label
      gold/minimal, and append to logs/router_training.jsonl
  (2) train:   run RouterTrainer (cost-ladder loss) -> store/router_model/

    python -m scripts.train --config config/file_router.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config              # noqa: E402
from file_router.memory import SharedEncoders, DatasetMemory  # noqa: E402
from file_router.router.zeroshot import ZeroShotRouter   # noqa: E402
from file_router.training import AutoLabeler, RouterTrainer, DataExporter  # noqa: E402
from file_router.utils import banner, dbg, info, warn    # noqa: E402


def _load_split_ids(cfg, ds, which):
    p = os.path.join(cfg.paths.splits_dir, f"{ds}.{which}.txt")
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return {l.strip() for l in f if l.strip()}


def _iter_qa(cfg, ds, keep_ids):
    p = os.path.join(cfg.paths.unified_dir, ds, "qa.jsonl")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["qa_id"] in keep_ids:
                yield rec


def collect(cfg, enc) -> int:
    """Recall + auto-label all train QA into router_training.jsonl."""
    # fresh training log each run
    if os.path.exists(cfg.paths.training_data_path):
        os.remove(cfg.paths.training_data_path)
    exporter = DataExporter(cfg.paths.training_data_path, compact=True)
    labeler = AutoLabeler(cfg)
    zeroshot = ZeroShotRouter(cfg)
    n_labeled = 0
    n_seen = 0
    started = time.perf_counter()
    for ds in cfg.datasets.enabled:        # only router-trainable datasets
        train_ids = _load_split_ids(cfg, ds, "train")
        if not train_ids:
            warn(f"[collect] {ds}: no train ids"); continue
        mem = DatasetMemory(cfg, ds, enc)
        retr = mem.retriever()
        info(f"--- collecting {ds} ({len(train_ids)} train QA) ---")
        for qa in _iter_qa(cfg, ds, train_ids):
            n_seen += 1
            groups, bias = retr.recall(qa["question"])
            if not groups:
                dbg(f"[collect] {qa['qa_id']} no candidates"); continue
            lab = labeler.label(
                qa, groups, mem.nodes, zeroshot=zeroshot, bias=bias,
            )
            if not lab:
                dbg(f"[collect] {qa['qa_id']} unlabelable"); continue
            # Write the labeled record once.  The old export-then-fill path
            # re-read and rewrote the entire growing JSONL for every question,
            # making collection O(N^2) and taking hours at large scale.
            qid = exporter.export(
                question=qa["question"], answer=qa["answer"],
                candidate_groups=groups, selected_groups=[], bias=bias,
                query_id=qa["qa_id"], gold_groups=lab["gold"],
                minimal_groups=lab["minimal"], correct=True,
            )
            n_labeled += 1
            if n_seen % 50 == 0:
                elapsed = time.perf_counter() - started
                info(f"[collect] progress={n_seen} labeled={n_labeled} "
                     f"elapsed={elapsed:.1f}s")
            dbg(f"[collect] {qa['qa_id']} -> {qid} "
                f"gold={len(lab['gold'])} min={len(lab['minimal'])} cand={len(groups)}")
    info(f"[collect] labeled {n_labeled}/{n_seen} train QA -> {cfg.paths.training_data_path}")
    return n_labeled


def fit_router_models(cfg) -> list[str]:
    """Train the proposed model and, when requested, a same-seed no-OPD model."""
    model_dirs = []
    opd_cfg = getattr(cfg.trainer, "opd", None)
    opd_enabled = bool(getattr(opd_cfg, "enabled", False))
    run_ab = opd_enabled and bool(getattr(opd_cfg, "run_ab", False))
    if run_ab:
        baseline_cfg = copy.deepcopy(cfg)
        baseline_cfg.trainer.opd.enabled = False
        baseline_cfg.trainer.lambda_opd = 0.0
        baseline_dir = cfg.paths.router_model_dir + "_baseline"
        banner("TRAIN A/B: baseline router (OPD disabled)")
        RouterTrainer(baseline_cfg).train(
            cfg.paths.training_data_path, baseline_dir)
        model_dirs.append(baseline_dir)

    banner("TRAIN: fit router MLP (cost-ladder + optional OPD loss)")
    RouterTrainer(cfg).train(
        cfg.paths.training_data_path, cfg.paths.router_model_dir)
    model_dirs.append(cfg.paths.router_model_dir)
    return model_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/file_router.yaml")
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    banner("TRAIN: collect candidates + auto-label")
    enc = SharedEncoders(cfg)
    if not args.skip_collect:
        n = collect(cfg, enc)
        if n == 0:
            warn("[train] no labeled data collected; aborting training.")
            sys.exit(2)

    fit_router_models(cfg)
    info("train stage done.")


if __name__ == "__main__":
    main()
