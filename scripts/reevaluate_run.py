"""Re-evaluate an existing validation run after inference/evaluator changes.

This reuses the source run's evidence stores and trained Router weights. It is
intended for cheap selector ablations: no download, ingest, labeling, or
training is repeated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _run_and_log(command: list[str], log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True,
                        help="completed validation_* directory to reuse")
    parser.add_argument("--output-root", default="",
                        help="default: the source run's parent directory")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    source = Path(args.source_run).expanduser().resolve()
    config_path = source / "effective_config.yaml"
    required = [
        config_path,
        source / "router_model" / "router_weights.pt",
        source / "router_model" / "router_meta.json",
        source / "router_model_baseline" / "router_weights.pt",
        source / "router_model_baseline" / "router_meta.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("source run is incomplete; missing: " + ", ".join(missing))

    output_root = (Path(args.output_root).expanduser().resolve()
                   if args.output_root else source.parent)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"reevaluation_selector_v2_{timestamp}"
    output = output_root / run_name
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "effective_config.source.yaml")
    with open(config_path, encoding="utf-8") as handle:
        effective = yaml.safe_load(handle)
    # Make model/training diagnostics follow the supplied source directory even
    # if that run was moved after download or upload.
    effective["paths"]["router_model_dir"] = str(source / "router_model")
    effective["paths"]["training_data_path"] = str(
        source / "router_training.jsonl")
    effective["paths"]["log_dir"] = str(source / "internal_logs")
    effective["router"]["router_model_path"] = ""
    reevaluation_config = output / "effective_config.yaml"
    with open(reevaluation_config, "w", encoding="utf-8") as handle:
        yaml.safe_dump(effective, handle, allow_unicode=True, sort_keys=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "type": "selector_reevaluation",
        "source_run": str(source),
        "status": "running",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [
        sys.executable, "-m", "scripts.evaluate_validation",
        "--config", str(reevaluation_config), "--run-dir", str(output),
    ]
    code = _run_and_log(command, output / "run.log")
    manifest["status"] = "complete" if code == 0 else "failed"
    manifest["exit_code"] = code
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if code == 0:
        print(f"\nRe-evaluation complete\nReport : {output / 'REPORT.md'}"
              f"\nRun dir: {output}")
    else:
        print(f"\nRe-evaluation failed; see {output / 'run.log'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
