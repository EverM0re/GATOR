"""Cross-dataset generalization: train the router on one dataset, test on others.

The question is whether the learned policy transfers, or whether it memorises
one corpus's layout.  A router that only works on the corpus it was trained on
is not a reusable component, which is the claim the mounting experiments make.

Produces a train x test matrix.  The diagonal is in-domain; off-diagonal cells
are transfer.  The honest comparison for each cell is against the training-free
zeroshot router on that same test set, since that is what a practitioner would
use without training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _run(command: List[str], log: Path) -> int:
    print(f"[gen] $ {' '.join(command)}")
    with open(log, "a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            handle.write(line)
        return process.wait()


def _effective_config(base_config: Path, datasets: List[str],
                      out_path: Path, model_dir: Path, epochs: int,
                      eval_per_dataset: int = 0, subset: int = 0,
                      docs: int = 0) -> None:
    """Write a config whose `enabled` list is exactly `datasets`.

    `enabled` drives BOTH training and evaluation here, so the train phase and
    each eval phase need their own config with a single-purpose list.  Passing
    train+test together (an earlier bug) silently evaluated on the union, which
    made every off-diagonal cell of the transfer matrix identical.
    """
    import yaml

    with open(base_config, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["datasets"]["enabled"] = list(dict.fromkeys(datasets))
    # These were previously declared but never applied, so runs silently used
    # whatever subset happened to be on disk -- producing 4-question "results".
    if subset:
        raw["datasets"]["subset_qa"] = subset
    if docs:
        raw["datasets"]["subset_docs"] = docs
    raw["trainer"]["num_epochs"] = epochs
    if eval_per_dataset:
        raw["evaluation"]["max_questions"] = eval_per_dataset
    raw["paths"]["router_model_dir"] = str(model_dir)
    raw["router"]["router_model_path"] = str(model_dir)
    out_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/file_router.yaml")
    parser.add_argument("--datasets", nargs="+",
                        default=["unidoc", "mmdocrag"],
                        help="datasets to cross; each is used as a train source")
    parser.add_argument("--out-root", default="experiment_campaigns/generalization")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-per-dataset", type=int, default=50)
    parser.add_argument("--subset", type=int, default=300)
    parser.add_argument("--docs", type=int, default=80)
    parser.add_argument("--skip-ingest", action="store_true",
                        help="reuse existing stores (the corpus is shared)")
    args = parser.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    log = root / "generalization.log"
    results: Dict[str, Dict[str, Any]] = {}

    for train_on in args.datasets:
        run_dir = root / f"train_{train_on}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_dir = run_dir / "router_model"
        config_path = run_dir / "config.yaml"
        # Train on this dataset ONLY.
        _effective_config(Path(args.config), [train_on],
                          config_path, model_dir, args.epochs,
                          subset=args.subset, docs=args.docs)

        env = dict(os.environ)
        env.update({"SKIP_DOWNLOAD": "1",
                    "SKIP_INGEST": "1" if args.skip_ingest else "0",
                    "SKIP_LLM_CHECK": "1"})

        # One model per training source, then evaluate it on every dataset.
        code = _run([sys.executable, "-m", "scripts.train",
                     "--config", str(config_path)], log)
        if code != 0:
            print(f"[gen] training on {train_on} failed (exit {code}); skipping")
            continue

        for test_on in args.datasets:
            eval_dir = run_dir / f"test_{test_on}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_config = eval_dir / "config.yaml"
            # Evaluate on the TEST dataset only; the model comes from model_dir.
            _effective_config(Path(args.config), [test_on],
                              eval_config, model_dir, args.epochs,
                              args.eval_per_dataset,
                              subset=args.subset, docs=args.docs)
            code = _run([sys.executable, "-m", "scripts.evaluate_validation",
                         "--config", str(eval_config),
                         "--run-dir", str(eval_dir)], log)
            summary_path = eval_dir / "summary.json"
            if code != 0 or not summary_path.exists():
                print(f"[gen] eval {train_on}->{test_on} failed")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            overall = summary.get("routers", {}).get("overall", {})
            n = overall.get("learned", {}).get("n") or 0
            if n < 20:
                print(f"[gen] WARNING: {train_on}->{test_on} evaluated only {n} "
                      "questions; treat this cell as noise, not a result")
            results.setdefault(train_on, {})[test_on] = {
                "n": n,
                "learned_f1": overall.get("learned", {}).get("f1_valid"),
                "learned_judge": overall.get("learned", {}).get("judge_accuracy"),
                "learned_cost": overall.get("learned", {}).get("avg_cost"),
                "learned_f1_per_1k": overall.get("learned", {}).get("f1_per_1k_tokens"),
                "learned_gold_page": overall.get("learned", {}).get(
                    "gold_page_any_selection_rate"),
                # Training-free reference on the same test set.
                "zeroshot_f1": overall.get("zeroshot", {}).get("f1_valid"),
                "zeroshot_f1_per_1k": overall.get("zeroshot", {}).get(
                    "f1_per_1k_tokens"),
            }

    (root / "generalization.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[gen] wrote {root / 'generalization.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
